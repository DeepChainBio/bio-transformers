"""
This script defines a parent class for transformers, for which child classes which are
specific to a given transformers implementation can inherit.
It allows to derive probabilities, embeddings and log-likelihoods based on inputs
sequences, and displays some properties of the transformer model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Tuple

import torch
import torch.tensor

from .utils import (
    TransformersModelProperties,
    _check_sequence,
)

NATURAL_AAS = "ACDEFGHIKLMNPQRSTVWY"
NATURAL_AAS_LIST = [AA for AA in NATURAL_AAS]


class TransformersWrapper(ABC):
    """
    Abstract class that uses pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(
        self,
        model_dir: str,
        _device: str = None,
        vocab_token_list: List[str] = None,
        mask_bool: bool = False,
    ):
        """Initialize Transformers wrapper

        Args:
            model_dir (str): name directory of the pretrained model
            _device (str, optional): type of device to use (cpu or cuda).
            vocab_token_list (List[str], optional): . Defaults to list(NATURAL_AAS).
            mask_bool (bool, optional): Wether to use mask or not for inference.
        """

        if _device is not None:
            print("Requested device: ", _device)
            if "cuda" in _device:
                try:
                    assert torch.cuda.is_available()
                except AssertionError:
                    print("No GPU available")
                _device = "cpu"
        else:
            _device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(_device)

        self.model_dir = model_dir
        self.mask_bool = mask_bool
        self.vocab_token_list = (
            list(NATURAL_AAS) if vocab_token_list is None else vocab_token_list
        )

    @property
    def model_id(self) -> str:
        """Model ID, as specified in the model directory"""
        return self.model_dir.lower()

    @property
    @abstractmethod
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""

    @property
    @abstractmethod
    def model_property(self) -> TransformersModelProperties:
        """Returns a class with model properties"""

    @property
    @abstractmethod
    def model_vocab_tokens(self) -> List[str]:
        """List of all vocabulary tokens to consider (as strings), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""

    @property
    @abstractmethod
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list"""

    @property
    @abstractmethod
    def model_vocab_ids(self) -> List[int]:
        """List of all vocabulary IDs to consider (as ints), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""

    @property
    @abstractmethod
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""

    @property
    @abstractmethod
    def pad_token(self) -> str:
        """Representation of the pad token (as a string)"""

    @property
    @abstractmethod
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string).
        Returns an empty string if no such token"""

    @property
    @abstractmethod
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string).
        Returns an empty string if no such token."""

    @property
    @abstractmethod
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""

    @abstractmethod
    def _process_sequences_and_tokens(
        self, sequences_list: List[str], tokens_list: List[str]
    ) -> Tuple[Dict[str, torch.tensor], torch.tensor, List[int]]:
        """Function to transform tokens string to IDs; it depends on the model used"""
        return NotImplemented, NotImplemented, NotImplemented

    @abstractmethod
    def _model_evaluation(
        self, model_inputs: Dict[str, torch.tensor], batch_size: int = 1,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Function which computes logits and embeddings based on a list of sequences"""
        return NotImplemented, NotImplemented

    def _generate_chunks(self, sequences_list: List[str], batch_size: int) -> List[str]:
        """Build a generator to yield protein sequences batch"""
        for i in range(0, len(sequences_list), batch_size):
            yield sequences_list[i : (i + batch_size)]

    def _generate_dict_chunks(
        self, sequence_dict: Dict[str, Any], batch_size: int
    ) -> Dict[str, Iterable]:
        """Yield a dictionnary of tensor"""
        first_key = list(sequence_dict.keys())[0]
        len_sequence = len(sequence_dict[first_key])
        for i in range(0, len_sequence, batch_size):
            batch_sequence = {
                key: value[i : (i + batch_size)] for key, value in sequence_dict.items()
            }
            yield batch_sequence

    def _repeat_and_mask_inputs(
        self, model_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], List[List]]:
        """Function that takes input tensor and create new one by masking and repeat"""
        new_input_ids = []
        new_attention_mask = []
        new_token_type_ids = []
        mask_ids = []
        for sequence, binary_mask, zeros in zip(
            model_inputs["input_ids"],
            model_inputs["attention_mask"],
            model_inputs["token_type_ids"],
        ):
            mask_id = []
            for i in range(1, sum(binary_mask) - 1):
                mask_sequence = torch.tensor(
                    sequence[:i].tolist()
                    + [self.token_to_id(self.mask_token)]
                    + sequence[i + 1 :].tolist(),
                    dtype=torch.int64,
                )
                new_input_ids.append(mask_sequence)
                new_attention_mask.append(binary_mask)
                new_token_type_ids.append(zeros)
                mask_id.append(i)
            mask_ids.append(mask_id)
        model_inputs["input_ids"] = torch.stack(new_input_ids)
        model_inputs["attention_mask"] = torch.stack(new_attention_mask)
        model_inputs["token_type_ids"] = torch.stack(new_token_type_ids)
        return model_inputs, mask_ids

    def _gather_masked_outputs(self, model_outputs, masked_ids_list):
        """Function that gathers all the masked outputs to original tensor shape"""
        max_length = model_outputs.shape[1]
        inf_tensor = -float("Inf") * torch.ones(
            [1, model_outputs.shape[2]], dtype=torch.float32
        )
        sequences_list = []
        start_id = 0
        for mask_id in masked_ids_list:
            end_id = start_id + len(mask_id)
            sequence = torch.cat(
                (
                    inf_tensor,
                    model_outputs[range(start_id, end_id), mask_id],
                    inf_tensor.repeat(max_length - len(mask_id) - 1, 1),
                ),
                0,
            )
            sequences_list.append(sequence)
            start_id = end_id
        return torch.stack(sequences_list)

    def _labels_remapping(self, labels, tokens):
        """Function that remaps IDs of the considered tokens from 0 to len(tokens)"""
        mapping = dict(zip(tokens, range(len(tokens))))
        return torch.tensor([mapping[lbl.item()] for lbl in labels])

    def _filter_logits(
        self, logits: torch.Tensor, labels: torch.Tensor, tokens=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Function to process logits by removing unconsidered tokens"""
        mask_filter = torch.zeros(labels.shape, dtype=torch.bool)
        for token_id in tokens:
            mask_filter += labels == token_id
        return (
            logits[mask_filter][:, tokens],
            self._labels_remapping(labels[mask_filter], tokens),
        )

    def _filter_loglikelihoods(
        self, logits: torch.Tensor, labels: torch.Tensor, tokens=None
    ) -> torch.Tensor:
        """Function to process loglikelihoods by removing unconsidered tokens"""
        log_softmax = torch.nn.LogSoftmax(dim=1)

        mask_filter = torch.zeros(labels.shape, dtype=torch.bool)
        for token_id in tokens:
            mask_filter += labels == token_id

        labels_list = [
            self._labels_remapping(lbl[fltr], tokens)
            for lbl, fltr in zip(labels, mask_filter)
        ]
        logprobs_list = [
            log_softmax(lgt[fltr][:, tokens]) for lgt, fltr in zip(logits, mask_filter)
        ]
        return torch.stack(
            [
                torch.sum(lgp[range(lbl.shape[0]), lbl])
                for lgp, lbl in zip(logprobs_list, labels_list)
            ]
        )

    def _filter_and_pool_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        pool_mode: str = "cls",
        tokens=None,
    ) -> torch.Tensor:
        """Function to process embeddings by removing unconsidered tokens and pooling"""

        if pool_mode == "cls":
            embeddings = embeddings[:, 0, :]
        else:
            # tokens filtering
            mask_filter = torch.zeros(labels.shape, dtype=torch.bool)
            for token_id in tokens:
                mask_filter += labels == token_id
            embeddings_list = [seq[msk] for seq, msk in zip(embeddings, mask_filter)]

            # embeddings pooling
            if pool_mode == "mean":
                embeddings = torch.stack(
                    [torch.mean(emb.float(), axis=0) for emb in embeddings_list]
                )
            elif pool_mode == "max":
                embeddings = torch.stack(
                    [torch.max(emb.float(), 0)[0] for emb in embeddings_list]
                )
            elif pool_mode == "min":
                embeddings = torch.stack(
                    [torch.min(emb.float(), 0)[0] for emb in embeddings_list]
                )

        return embeddings

    def _compute_logits(self, model_inputs, batch_size, pass_mode):
        """Intermediate function to compute logits"""
        if pass_mode == "masked":
            model_inputs, masked_ids_list = self._repeat_and_mask_inputs(model_inputs)
            logits, _ = self._model_evaluation(model_inputs, batch_size=batch_size)
            logits = self._gather_masked_outputs(logits, masked_ids_list)
        elif pass_mode == "forward":
            logits, _ = self._model_evaluation(model_inputs, batch_size=batch_size)
        return logits

    def _compute_embeddings(self, model_inputs, batch_size):
        """Intermediate function to compute embeddings"""
        _, embeddings = self._model_evaluation(model_inputs, batch_size=batch_size)
        return embeddings

    def compute_logits(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        pass_mode: str = "forward",
        tokens_list: List[str] = NATURAL_AAS_LIST,
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Function that computes the logits from sequences

        Args:
            sequences_list (List[str]): List of sequences
            batch_size (int, optional): Batch size
            pass_mode (str, optional): Mode of model evaluation ('forward' or 'masked')
            tokens_list (List[str], optional): List of tokens to consider

        Returns:
            Tuple[torch.tensor, torch.tensor]: logits and labels in torch.tensor format
        """
        _check_sequence(sequences_list, self.model_dir, 1024)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )

        logits = self._compute_logits(inputs, batch_size, pass_mode)

        logits, labels = self._filter_logits(logits, labels, tokens)

        return logits, labels

    def compute_loglikelihoods(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        pass_mode: str = "forward",
        tokens_list: List[str] = NATURAL_AAS_LIST,
    ) -> torch.Tensor:
        """Function that computes loglikelihoods of sequences

        Args:
            sequences_list (List[str]): List of sequences
            batch_size (int, optional): Batch size
            pass_mode (str, optional): Mode of model evaluation ('forward' or 'masked')
            tokens_list (List[str], optional): List of tokens to consider

        Returns:
            torch.Tensor: loglikelihoods in torch.tensor format
        """

        _check_sequence(sequences_list, self.model_dir, 1024)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )

        logits = self._compute_logits(inputs, batch_size, pass_mode)

        loglikelihoods = self._filter_loglikelihoods(logits, labels, tokens)

        return loglikelihoods

    def compute_embeddings(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        pool_mode: str = "cls",
        tokens_list: List[str] = NATURAL_AAS_LIST,
    ) -> torch.Tensor:
        """Function that computes embeddings of sequences

        Args:
            sequences_list (List[str]): List of sequences
            batch_size (int, optional): Batch size
            pool_mode (str, optional): Mode of pooling ('cls', 'mean', 'max' or 'min')
            tokens_list (List[str], optional): List of tokens to consider

        Returns:
            torch.Tensor: Tensor of shape [number_of_sequences, embeddings_size]
        """
        _check_sequence(sequences_list, self.model_dir, 1024)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )

        embeddings = self._compute_embeddings(inputs, batch_size)

        embeddings = self._filter_and_pool_embeddings(
            embeddings, labels, pool_mode, tokens
        )

        return embeddings
