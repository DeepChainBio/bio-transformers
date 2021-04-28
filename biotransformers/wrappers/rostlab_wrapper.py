"""
This script defines a class which inherits from the TransformersWrapper class, and is
specific to the Rostlab models (eg ProtBert and ProtBert-BFD) developed by
hugging face
- ProtBert: https://huggingface.co/Rostlab/prot_bert
- ProtBert BFD: https://huggingface.co/Rostlab/prot_bert_bfd
"""
from typing import Dict, List, Tuple

import torch
from biotransformers.wrappers.transformers_wrappers import (
    TransformersModelProperties,
    TransformersWrapper,
)
from torch.nn import DataParallel
from transformers import BertForMaskedLM, BertTokenizer

rostlab_list = ["Rostlab/prot_bert", "Rostlab/prot_bert_bfd"]
DEFAULT_MODEL = "Rostlab/prot_bert"


class RostlabWrapper(TransformersWrapper):
    """
    Class that uses a rostlab type of pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(self, model_dir: str, device, multi_gpu):

        if model_dir not in rostlab_list:
            print(
                f"Model dir '{model_dir}' not recognized. "
                f"Using '{DEFAULT_MODEL}' as default"
            )
            model_dir = DEFAULT_MODEL

        super().__init__(model_dir, _device=device, multi_gpu=multi_gpu)

        self.tokenizer = BertTokenizer.from_pretrained(
            model_dir, do_lower_case=False, padding=True
        )
        self.model_dir = model_dir
        self.model = (
            BertForMaskedLM.from_pretrained(self.model_dir).eval().to(self._device)
        )
        self.hidden_size = self.model.config.hidden_size
        if self.multi_gpu:
            self.model = DataParallel(self.model)

        self.mask_pipeline = None

    @property
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""
        return self.model_id.replace("rostlab/", "")

    @property
    def model_property(self) -> TransformersModelProperties:
        """Returns a class with model properties"""
        return TransformersModelProperties(
            num_sep_tokens=2, begin_token=True, end_token=True
        )

    @property
    def model_vocab_tokens(self) -> List[str]:
        """List of all vocabulary tokens to consider (as strings), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""
        voc = (
            self.vocab_token_list
            if self.vocab_token_list is not None
            else self.tokenizer.vocab
        )
        return voc

    @property
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list"""
        return list(self.tokenizer.vocab.keys())

    @property
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size"""
        return len(list(self.tokenizer.vocab.keys()))

    @property
    def model_vocab_ids(self) -> List[int]:
        """List of all vocabulary IDs to consider (as ints), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""
        vocab_ids = [
            self.tokenizer.convert_tokens_to_ids(tok) for tok in self.model_vocab_tokens
        ]
        return vocab_ids

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
        return "[CLS]"

    @property
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string)."""
        return "[SEP]"

    @property
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""
        return lambda x: self.tokenizer.convert_tokens_to_ids(x)

    @property
    def embeddings_size(self) -> int:
        """Returns size of the embeddings"""
        return self.hidden_size

    def _process_sequences_and_tokens(
        self, sequences_list: List[str], tokens_list: List[str]
    ) -> Tuple[Dict[str, torch.tensor], torch.tensor, List[int]]:
        """Function to transform tokens string to IDs; it depends on the model used"""
        tokens = []
        for token in tokens_list:
            if token not in self.model_vocabulary:
                print("Warnings; token", token, "does not belong to model vocabulary")
            else:
                tokens.append(self.token_to_id(token))

        separated_sequences_list = [" ".join(seq) for seq in sequences_list]
        encoded_inputs = self.tokenizer(
            separated_sequences_list,
            return_tensors="pt",
            padding=True,
        ).to("cpu")

        return encoded_inputs, encoded_inputs["input_ids"], tokens

    def _model_pass(
        self, model_inputs: Dict[str, torch.tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function which computes logits and embeddings based on a dict of sequences
        tensors, a provided batch size and an inference configuration. The output is
        obtained by computing a forward pass through the model ("forward inference")

        Args:
            model_inputs (Dict[str, torch.tensor]): [description]

        Returns:
            Tuple[torch.tensor, torch.tensor]:
                    * logits [num_seqs, max_len_seqs, vocab_size]
                    * embeddings [num_seqs, max_len_seqs+1, embedding_size]
        """
        with torch.no_grad():
            model_inputs = {
                key: value.to(self._device) for key, value in model_inputs.items()
            }
            model_outputs = self.model(**model_inputs, output_hidden_states=True)
            logits = model_outputs.logits.detach().cpu()
            embeddings = model_outputs.hidden_states[-1].detach().cpu()

        return logits, embeddings
