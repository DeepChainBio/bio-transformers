"""
This script defines a class which inherits from the TransformersWrapper class, and is
specific to the Rostlab models (eg ProtBert and ProtBert-BFD) developed by
hugging face
- ProtBert: https://huggingface.co/Rostlab/prot_bert
- ProtBert BFD: https://huggingface.co/Rostlab/prot_bert_bfd
"""
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm
from transformers import BertForMaskedLM, BertTokenizer
from torch.nn import DataParallel

from .transformers_wrappers import (
    TransformersModelProperties,
    TransformersWrapper,
)

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
            separated_sequences_list, return_tensors="pt", padding=True,
        ).to(self._device)
        return encoded_inputs, encoded_inputs["input_ids"].to("cpu"), tokens

    def _model_evaluation(
        self, model_inputs: Dict[str, torch.tensor], batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by computing a forward pass through the model ("forward inference")

        Args:
            model_inputs (Dict[str, torch.tensor]): [description]
            batch_size (int): [description]

        Returns:
            Tuple[torch.tensor, torch.tensor]:
                - Logits: [num_seqs, max_len_seqs+2, vocab_size]
                - Embeddings: [num_seqs, max_len_seqs+2, embedding_size]
        """
        num_of_sequences = model_inputs["input_ids"].shape[0]
        num_batch_iter = int(np.ceil(num_of_sequences / batch_size))
        logits = torch.Tensor()  # [num_seqs, max_len_seqs+2, vocab_size]
        embeddings = torch.Tensor()  # [num_seqs, max_len_seqs+2, embedding_size]

        for batch_encoded in tqdm(
            self._generate_dict_chunks(model_inputs, batch_size), total=num_batch_iter
        ):
            with torch.no_grad():
                output = self.model(**batch_encoded, output_hidden_states=True)
            new_logits = output.logits.detach().cpu()
            logits = torch.cat((logits, new_logits), dim=0)
            # Only keep track of the hidden states of the last layer
            new_embeddings = output.hidden_states[-1]

            new_embeddings = new_embeddings.detach().cpu()
            embeddings = torch.cat((embeddings, new_embeddings), dim=0)

        return logits, embeddings
