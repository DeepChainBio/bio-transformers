"""
This script defines a class which inherits from the LanguageModel class, and is
specific to the Rostlab models (eg ProtBert and ProtBert-BFD) developed by
hugging face
- ProtBert: https://huggingface.co/Rostlab/prot_bert
- ProtBert BFD: https://huggingface.co/Rostlab/prot_bert_bfd
"""
from typing import Dict, List, Tuple

import torch
import copy
from biotransformers.lightning_utils.data import (
    AlphabetDataLoader,
    convert_ckpt_to_statedict,
)
from biotransformers.utils.constant import DEFAULT_ROSTLAB_MODEL, ROSTLAB_LIST
from biotransformers.utils.logger import logger  # noqa
from biotransformers.utils.utils import _generate_chunks, _get_num_batch_iter
from biotransformers.wrappers.language_model import LanguageModel
from ray.actor import ActorHandle
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer

log = logger("rostlab_wrapper")


class RostlabWrapper(LanguageModel):
    """
    Class that uses a rostlab type of pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(self, model_dir: str, device):
        if model_dir not in ROSTLAB_LIST:
            print(
                f"Model dir '{model_dir}' not recognized."
                f" Using '{DEFAULT_ROSTLAB_MODEL}' as default"
            )
            model_dir = DEFAULT_ROSTLAB_MODEL
        super().__init__(model_dir=model_dir, device=device)
        self.tokenizer = BertTokenizer.from_pretrained(
            model_dir, do_lower_case=False, padding=True
        )
        self._model = (
            BertForMaskedLM.from_pretrained(self._model_dir).eval().to(self._device)
        )
        self.hidden_size = self._model.config.hidden_size
        self.mask_pipeline = None

    @property
    def model(self) -> torch.nn.Module:
        """Return torch model."""
        return self._model

    @property
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""
        return self.model_id.replace("rostlab/", "")

    @property
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list"""
        return list(self.tokenizer.vocab.keys())

    @property
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size"""
        return self.tokenizer.vocab_size

    @property
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""
        return self.tokenizer.mask_token  # "[MASK]"

    @property
    def pad_token(self) -> str:
        """Representation of the pad token (as a string)"""
        return self.tokenizer.pad_token  # "[PAD]"

    @property
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string)"""
        return self.tokenizer.cls_token  # "[CLS]"

    @property
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string)."""
        return self.tokenizer.sep_token  # "[SEP]"

    @property
    def does_end_token_exist(self) -> bool:
        """Returns true if a end of sequence token exists"""
        return True

    @property
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""
        return lambda x: self.tokenizer.convert_tokens_to_ids(x)

    @property
    def embeddings_size(self) -> int:
        """Returns size of the embeddings"""
        return self.hidden_size

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

    def process_sequences_and_tokens(
        self,
        sequences_list: List[str],
    ) -> Dict[str, torch.tensor]:
        """Function to transform tokens string to IDs; it depends on the model used"""
        separated_sequences_list = [" ".join(seq) for seq in sequences_list]
        encoded_inputs = self.tokenizer(
            separated_sequences_list,
            return_tensors="pt",
            padding=True,
        ).to("cpu")
        return encoded_inputs

    def model_pass(
        self,
        model_inputs: Dict[str, torch.tensor],
        batch_size: int,
        silent: bool = False,
        pba: ActorHandle = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function which computes logits and embeddings based on a dict of sequences
        tensors, a provided batch size and an inference configuration. The output is
        obtained by computing a forward pass through the model ("forward inference")

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
                model_inputs = {
                    key: value.to(self._device) for key, value in batch_inputs.items()
                }
                model_outputs = self._model(**model_inputs, output_hidden_states=True)
                batch_logits = model_outputs.logits.detach().cpu()
                batch_embeddings = model_outputs.hidden_states[-1].detach().cpu()
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
            x_ = [" ".join(seq) for seq in x]
            tokens = self.tokenizer(x_, return_tensors="pt", padding=True)
            return x, tokens["input_ids"]

        all_tokens = copy.deepcopy(self.tokenizer.vocab)
        del all_tokens["[PAD]"]
        del all_tokens["[UNK]"]
        del all_tokens["[CLS]"]
        del all_tokens["[SEP]"]
        del all_tokens["[MASK]"]
        standard_tokens = list(all_tokens.keys())

        alphabet_dl = AlphabetDataLoader(
            prepend_bos=True,
            append_eos=True,
            mask_idx=self.tokenizer.mask_token_id,
            pad_idx=self.tokenizer.pad_token_id,
            standard_toks=standard_tokens,
            model_dir=self._model_dir,
            lambda_toks_to_ids=lambda x: self.tokenizer.convert_tokens_to_ids(x),
            lambda_tokenizer=lambda x: tokenize(x),
        )
        return alphabet_dl
