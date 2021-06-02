"""
This script defines a class which inherits from the TransformersWrapper class, and is
specific to the ESM model developed by FAIR (https://github.com/facebookresearch/esm).
"""

from typing import Dict, List, Tuple

import esm
import torch
from biotransformers.lightning_utils.data import AlphabetDataLoader
from biotransformers.utils.constant import DEFAULT_ESM_MODEL, ESM_LIST
from biotransformers.utils.logger import logger  # noqa
from biotransformers.wrappers.transformers_wrappers import TransformersWrapper
from torch.nn import DataParallel

log = logger("esm_wrapper")


class ESMWrapper(TransformersWrapper):
    """
    Class that uses an ESM type of pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(self, model_dir: str, device, multi_gpu):

        if model_dir not in ESM_LIST:
            print(
                f"Model dir '{model_dir}' not recognized. "
                f"Using '{DEFAULT_ESM_MODEL}' as default"
            )
            model_dir = DEFAULT_ESM_MODEL

        super().__init__(model_dir, _device=device, multi_gpu=multi_gpu)

        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_dir)
        self.num_layers = self.model.num_layers
        repr_layers = -1
        self.repr_layers = (repr_layers + self.num_layers + 1) % (self.num_layers + 1)
        self.hidden_size = self.model.args.embed_dim

        if self.multi_gpu:
            self.model = DataParallel(self.model).to(self._device)
        else:
            self.model = self.model.to(self._device)
        self.batch_converter = self.alphabet.get_batch_converter()

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

    def _process_sequences_and_tokens(
        self, sequences_list: List[str], tokens_list: List[str]
    ) -> Tuple[Dict[str, torch.tensor], torch.tensor, List[int]]:
        """Function to transform tokens string to IDs; it depends on the model used"""
        tokens = []
        for token in tokens_list:
            if token not in self.model_vocabulary:
                log.warning("token %s does not belong to model vocabulary" % token)
            else:
                tokens.append(self.token_to_id(token))

        _, _, all_tokens = self.batch_converter(
            [("", sequence) for sequence in sequences_list]
        )

        all_tokens = all_tokens.to("cpu")

        encoded_inputs = {
            "input_ids": all_tokens,
            "attention_mask": 1 * (all_tokens != self.token_to_id(self.pad_token)),
            "token_type_ids": torch.zeros(all_tokens.shape),
        }
        return encoded_inputs, all_tokens, tokens

    def _model_pass(
        self, model_inputs: Dict[str, torch.tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by computing a forward pass through the model ("forward inference")

        Args:
            model_inputs (Dict[str, torch.tensor]): [description]

        Returns:
            Tuple[torch.tensor, torch.tensor]:
                    * logits [num_seqs, max_len_seqs, vocab_size]
                    * embeddings [num_seqs, max_len_seqs+1, embedding_size]
        """
        with torch.no_grad():
            model_outputs = self.model(
                model_inputs["input_ids"].to(self._device),
                repr_layers=[self.repr_layers],
            )
            logits = model_outputs["logits"].detach().cpu()
            embeddings = (
                model_outputs["representations"][self.repr_layers].detach().cpu()
            )

        return logits, embeddings

    def _get_alphabet_dataloader(self):
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
            lambda_toks_to_ids=lambda x: self.alphabet.tok_to_idx[x],
            lambda_tokenizer=lambda x: tokenize(x),
        )

        return alphabet_dl
