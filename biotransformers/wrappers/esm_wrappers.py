"""
This script defines a class which inherits from the LanguageModel class, and is
specific to the ESM model developed by FAIR (https://github.com/facebookresearch/esm).
"""

from typing import Dict, List, Tuple

import esm
import torch
from biotransformers.lightning_utils.data import (
    AlphabetDataLoader,
    convert_ckpt_to_statedict,
)
from biotransformers.utils.constant import DEFAULT_ESM_MODEL, ESM_LIST
from biotransformers.utils.logger import logger  # noqa
from biotransformers.utils.utils import _generate_chunks, _get_num_batch_iter
from biotransformers.wrappers.language_model import LanguageModel
from ray.actor import ActorHandle
from tqdm import tqdm

log = logger("esm_wrapper")
path_msa_folder = str


class ESMWrapper(LanguageModel):
    """
    Class that uses an ESM type of pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(self, model_dir: str, device: str):
        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL
        super().__init__(model_dir=model_dir, device=device)
        self._model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_dir)
        self.num_layers = self._model.num_layers
        repr_layers = -1
        self.repr_layers = (repr_layers + self.num_layers + 1) % (self.num_layers + 1)
        self.hidden_size = self._model.args.embed_dim
        self._model = self._model.to(self._device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.is_msa = "msa" in model_dir

    @property
    def model(self) -> torch.nn.Module:
        """Return torch model."""
        return self._model

    @property
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""
        return self.model_id

    @property
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list"""
        return list(self.alphabet.tok_to_idx.keys())

    @property
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size"""
        return len(list(self.alphabet.tok_to_idx.keys()))

    @property
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.mask_idx]  # "<mask>"

    @property
    def pad_token(self) -> str:
        """Representation of the pad token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.padding_idx]  # "<pad>"

    @property
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.cls_idx]  # "<cls>"

    @property
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string)"""
        return self.alphabet.all_toks[self.alphabet.eos_idx]  # "<eos>"

    @property
    def does_end_token_exist(self) -> bool:
        """Returns true if a end of sequence token exists"""
        return self.alphabet.append_eos

    @property
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""
        return lambda x: self.alphabet.tok_to_idx[x]

    @property
    def embeddings_size(self):
        """Returns size of the embeddings"""
        return self.hidden_size

    def process_sequences_and_tokens(
        self, sequences_list: List[str]
    ) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model used"""
        if self.is_msa:
            _, _, all_tokens = self.batch_converter(sequences_list)
        else:
            _, _, all_tokens = self.batch_converter(
                [("", sequence) for sequence in sequences_list]
            )

        all_tokens = all_tokens.to("cpu")
        encoded_inputs = {
            "input_ids": all_tokens,
            "attention_mask": 1 * (all_tokens != self.token_to_id(self.pad_token)),
            "token_type_ids": torch.zeros(all_tokens.shape),
        }
        return encoded_inputs

    def _load_model(self, path_model: str, map_location=None):
        """Load model."""
        if path_model.endswith(".pt"):
            loaded_model = torch.load(path_model)
        elif path_model.endswith(".ckpt"):
            loaded_model = convert_ckpt_to_statedict(
                torch.load(path_model)["state_dict"]
            )
        else:
            raise ValueError("Expecting a .pt or .ckpt file")
        self._model.load_state_dict(loaded_model, map_location)
        self._model.eval()
        log.info("Load model %s" % path_model)

    def model_pass(
        self,
        model_inputs: Dict[str, torch.Tensor],
        batch_size: int,
        silent: bool = False,
        pba: ActorHandle = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by computing a forward pass through the model ("forward inference")

        The datagenerator is not the same the multi_gpus inference. We use a tqdm progress bar
        that is updated by the worker. The progress bar is instantiated before ray.remote

        Args:
            model_inputs (Dict[str, torch.tensor]): [description]
            batch_size (int): size of the batch
            silent : display or not progress bar
            pba : tqdm progress bar for ray actor
        Returns:
            Tuple[torch.tensor, torch.tensor]:
                    * logits [num_seqs, max_len_seqs, vocab_size]
                    * embeddings [num_seqs, max_len_seqs+1, embedding_size]
        """
        if pba is None:
            batch_generator = tqdm(
                _generate_chunks(model_inputs, batch_size),
                total=_get_num_batch_iter(model_inputs, batch_size),
                disable=silent,
            )
        else:
            batch_generator = _generate_chunks(model_inputs, batch_size)

        logits = torch.Tensor()  # [num_seqs, max_len_seqs+1, vocab_size]
        embeddings = torch.Tensor()  # [num_seqs, max_len_seqs+1, embedding_size]
        for batch_inputs in batch_generator:
            with torch.no_grad():
                model_outputs = self._model(
                    batch_inputs["input_ids"].to(self._device),
                    repr_layers=[self.repr_layers],
                )
                batch_logits = model_outputs["logits"].detach().cpu()
                batch_embeddings = (
                    model_outputs["representations"][self.repr_layers].detach().cpu()
                )
                embeddings = torch.cat((embeddings, batch_embeddings), dim=0)
                logits = torch.cat((logits, batch_logits), dim=0)

            # tqdm worker update
            if pba is not None:
                pba.update.remote(1)
        return logits, embeddings

    def get_alphabet_dataloader(self):
        """Define an alphabet mapping for common method between
        protbert and ESM
        """

        def tokenize(x: List[str]):
            x_ = list(enumerate(x))
            _, seqs, tokens = self.batch_converter(x_)
            return seqs, tokens

        alphabet_dl = AlphabetDataLoader(
            prepend_bos=True,
            append_eos=True,
            mask_idx=self.alphabet.mask_idx,
            pad_idx=self.alphabet.padding_idx,
            standard_toks=self.alphabet.standard_toks,
            model_dir=self._model_dir,
            lambda_toks_to_ids=lambda x: self.alphabet.tok_to_idx[x],
            lambda_tokenizer=lambda x: tokenize(x),
        )

        return alphabet_dl
