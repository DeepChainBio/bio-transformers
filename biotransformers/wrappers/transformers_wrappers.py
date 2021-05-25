"""
This script defines a parent class for transformers, for which child classes which are
specific to a given transformers implementation can inherit.
It allows to derive probabilities, embeddings and log-likelihoods based on inputs
sequences, and displays some properties of the transformer model.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.tensor
from biotransformers.utils.gpus_utils import set_device
from biotransformers.utils.utils import (
    _check_memory_embeddings,
    _check_memory_logits,
    _check_sequence,
)
from torch.nn import functional as F  # noqa: N812
from tqdm import tqdm

NATURAL_AAS_LIST = list("ACDEFGHIKLMNPQRSTVWY")


class TransformersWrapper(ABC):
    """
    Abstract class that uses pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(
        self,
        model_dir: str,
        _device: str = None,
        multi_gpu: bool = False,
        vocab_token_list: List[str] = None,
        mask_bool: bool = False,
    ):
        """Initialize Transformers wrapper

        Args:
            model_dir: name directory of the pretrained model
            _device: type of device to use (cpu or cuda).
            vocab_token_list: Defaults to list(NATURAL_AAS).
            mask_bool: Wether to use mask or not for inference.
        """
        _device, _multi_gpu = set_device(_device, multi_gpu)

        self._device = torch.device(_device)
        self.multi_gpu = _multi_gpu
        self.model_dir = model_dir
        self.mask_bool = mask_bool

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
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list"""

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size"""

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
    def does_end_token_exist(self) -> bool:
        """Returns true if a end of sequence token exists"""

    @property
    @abstractmethod
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""

    @property
    @abstractmethod
    def embeddings_size(self) -> int:
        """Returns size of the embeddings"""

    @abstractmethod
    def _process_sequences_and_tokens(
        self, sequences_list: List[str], tokens_list: List[str]
    ) -> Tuple[Dict[str, torch.tensor], torch.tensor, List[int]]:
        """Function to transform tokens string to IDs; it depends on the model used"""
        return NotImplemented, NotImplemented, NotImplemented

    @abstractmethod
    def _model_pass(
        self, model_inputs: Dict[str, torch.tensor]
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Function which computes logits and embeddings based on a list of sequences"""
        return NotImplemented, NotImplemented

    def _get_num_batch_iter(self, model_inputs: Dict[str, Any], batch_size: int) -> int:
        num_of_sequences = model_inputs["input_ids"].shape[0]
        num_batch_iter = int(np.ceil(num_of_sequences / batch_size))
        return num_batch_iter

    def _generate_chunks(
        self, model_inputs: Dict[str, Any], batch_size: int
    ) -> Generator[Dict[str, Iterable], None, None]:
        """Yield a dictionnary of tensor"""
        num_of_sequences = model_inputs["input_ids"].shape[0]
        for i in range(0, num_of_sequences, batch_size):
            batch_sequence = {
                key: value[i : (i + batch_size)] for key, value in model_inputs.items()
            }
            yield batch_sequence

    def _repeat_and_mask_inputs(
        self, model_inputs: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], List[List]]:
        """Create new tensor by masking each token and repeating sequence

        Args:
            model_inputs: shape -> (num_seqs, max_seq_len)

        Returns:
            model_inputs: shape -> (sum_tokens, max_seq_len)
            masked_ids_list: len -> (num_seqs)
        """
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
            for i in range(1, sum(binary_mask) - self.does_end_token_exist * 1):
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

    def _gather_masked_outputs(
        self, model_outputs: torch.Tensor, masked_ids_list: List[List]
    ) -> torch.Tensor:
        """Gather all the masked outputs to get original tensor shape

        Args:
            model_outputs (torch.Tensor): shape -> (sum_tokens, max_seq_len, vocab_size)
            masked_ids_list (List[List]) : len -> (num_seqs)

        Returns:
            model_outputs (torch.Tensor): shape -> (num_seqs, max_seq_len, vocab_size)
        """
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

    def _labels_remapping(
        self, labels: torch.Tensor, tokens: List[int]
    ) -> torch.Tensor:
        """Function that remaps IDs of the considered tokens from 0 to len(tokens)"""
        mapping = dict(zip(tokens, range(len(tokens))))
        return torch.tensor([mapping[lbl.item()] for lbl in labels])

    def _label_remapping(self, label: int, tokens: List[int]) -> int:
        """Function that remaps IDs of the considered tokens from 0 to len(tokens)"""
        mapping = dict(zip(tokens, range(len(tokens))))
        return mapping[label]

    def _filter_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tokens: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Remove unconsidered tokens from sequences and logits

        Args:
            logits (torch.Tensor): shape -> (num_seqs, max_seq_len, vocab_size)
            labels (torch.Tensor): shape -> (num_seqs, max_seq_len)
            tokens (List[int]): len -> (num_considered_token)

        Returns:
            logits (torch.Tensor): shape -> (sum_considered_token, num_considered_token)
            labels (torch.Tensor): shape -> (sum_considered_token,)
        """
        mask_filter = torch.zeros(labels.shape, dtype=torch.bool)
        for token_id in tokens:
            mask_filter += labels == token_id
        return (
            logits[mask_filter][:, tokens],
            self._labels_remapping(labels[mask_filter], tokens),
        )

    def _filter_loglikelihoods(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tokens: List[int],
    ) -> torch.Tensor:
        """Function to compute the loglikelihood of sequences based on logits
        Args:
            logits : [description]
            labels : Position of
            tokens: [description]

        Returns:
            Torch.tensor: tensor
        """
        masks = torch.zeros(labels.shape, dtype=torch.bool)
        for token_id in tokens:
            masks += labels == token_id

        loglikelihoods = []
        log_softmax = torch.nn.LogSoftmax(dim=0)
        # loop over the sequences
        for sequence_logit, sequence_label, sequence_mask in zip(logits, labels, masks):
            if sum(sequence_mask) == 0:
                loglikelihood = torch.tensor(float("NaN"))
            else:
                loglikelihood = 0
                # loop over the tokens
                for logit, label, mask in zip(
                    sequence_logit, sequence_label, sequence_mask
                ):
                    if mask:
                        loglikelihood += log_softmax(logit[tokens])[
                            self._label_remapping(label.item(), tokens)
                        ]
            loglikelihoods.append(loglikelihood)
        return torch.stack(loglikelihoods)

    def _filter_and_pool_embeddings(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        tokens: List[int],
        pool_mode: Tuple[str, ...] = ("cls", "mean"),
    ) -> Dict[str, torch.Tensor]:
        """Remove unconsidered tokens from sequences and pool embeddings

        Args:
            logits (torch.Tensor): shape -> (num_seqs, max_seq_len, vocab_size)
            labels (torch.Tensor): shape -> (num_seqs, max_seq_len)
            tokens (List[int]): len -> (num_considered_token)
            pool_mode (Tuple[str]):

        Returns:
            embeddings[str] (torch.Tensor): shape -> (num_seqs, emb_size)
        """
        # cls pooling
        embeddings_dict = {}
        if "cls" in pool_mode:
            embeddings_dict["cls"] = embeddings[:, 0, :]

        # tokens filtering
        mask_filter = torch.zeros(labels.shape, dtype=torch.bool)
        for token_id in tokens:
            mask_filter += labels == token_id
        embeddings = [seq[msk] for seq, msk in zip(embeddings, mask_filter)]

        # embeddings pooling
        if "mean" in pool_mode:
            embeddings_dict["mean"] = torch.stack(
                [torch.mean(emb.float(), axis=0) for emb in embeddings]
            )
        if "max" in pool_mode:
            embeddings_dict["max"] = torch.stack(
                [torch.max(emb.float(), 0)[0] for emb in embeddings]
            )
        if "min" in pool_mode:
            embeddings_dict["min"] = torch.stack(
                [torch.min(emb.float(), 0)[0] for emb in embeddings]
            )

        if "full" in pool_mode:
            embeddings_dict["full"] = torch.stack([emb.float() for emb in embeddings])

        return embeddings_dict

    def _model_evaluation(
        self, model_inputs: Dict[str, torch.tensor], batch_size: int = 1, **kwargs
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
                    * logits [num_seqs, max_len_seqs, vocab_size]
                    * embeddings [num_seqs, max_len_seqs+1, embedding_size]
        """
        silent = kwargs.get("silent", False)
        # Initialize logits and embeddings before looping over batches
        logits = torch.Tensor()  # [num_seqs, max_len_seqs+1, vocab_size]
        embeddings = torch.Tensor()  # [num_seqs, max_len_seqs+1, embedding_size]

        for batch_inputs in tqdm(
            self._generate_chunks(model_inputs, batch_size),
            total=self._get_num_batch_iter(model_inputs, batch_size),
            disable=silent,
        ):
            batch_logits, batch_embeddings = self._model_pass(batch_inputs)

            embeddings = torch.cat((embeddings, batch_embeddings), dim=0)
            logits = torch.cat((logits, batch_logits), dim=0)

        return logits, embeddings

    def _compute_logits(
        self,
        model_inputs: Dict[str, torch.Tensor],
        batch_size: int,
        pass_mode: str,
        **kwargs
    ) -> torch.Tensor:
        """Intermediate function to compute logits

        Args:
            model_inputs[str] (torch.Tensor): shape -> (num_seqs, max_seq_len)
            batch_size (int)
            pass_mode (str)

        Returns:
            logits (torch.Tensor): shape -> (num_seqs, max_seq_len, vocab_size)
        """
        if pass_mode == "masked":
            model_inputs, masked_ids_list = self._repeat_and_mask_inputs(model_inputs)
            logits, _ = self._model_evaluation(
                model_inputs, batch_size=batch_size, **kwargs
            )
            logits = self._gather_masked_outputs(logits, masked_ids_list)
        elif pass_mode == "forward":
            logits, _ = self._model_evaluation(
                model_inputs, batch_size=batch_size, **kwargs
            )
        return logits

    def _compute_accuracy(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        """Intermediate function to compute accuracy

        Args:
            logits (torch.Tensor): shape -> (sum_considered_token, num_considered_token)
            labels (torch.Tensor): shape -> (sum_considered_token)

        Returns:
            accuracy (float)
        """
        softmaxes = F.softmax(logits, dim=1)
        _, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        return accuracies.float().mean().item()

    def _compute_calibration(
        self, logits: torch.Tensor, labels: torch.Tensor, n_bins: int = 10
    ) -> Dict[str, Any]:
        """Intermediate function to compute calibration

        Args:
            logits (torch.Tensor): shape -> (sum_considered_token, num_considered_token)
            labels (torch.Tensor): shape -> (sum_considered_token)
            n_bins (int)

        Returns:
            accuracy (float)
            ece (float)
            reliability_diagram (List[float])
        """
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        reliability_diagram = []
        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                reliability_diagram.append(accuracy_in_bin.item())
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            else:
                reliability_diagram.append(0.0)

        return {
            "accuracy": accuracies.float().mean().item(),
            "ece": ece.item(),
            "reliability_diagram": reliability_diagram,
        }

    def compute_logits(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        tokens_list: List[str] = None,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> Tuple[List[np.ndarray]]:
        """Function that computes the logits from sequences.

        It returns a list of logits for each sequence. Each sequence in the list
        contains only the amino acid to interest.

        Args:
            sequences_list: List of sequences
            batch_size: number of sequences to consider for the forward pass
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            tokens_list: List of tokens to consider

        Returns:
            Tuple[torch.tensor, torch.tensor]: logits and labels in torch.tensor format
        """
        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        _check_sequence(sequences_list, self.model_dir, 1024)
        _check_memory_logits(sequences_list, self.vocab_size, pass_mode)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        logits, labels = self._filter_logits(logits, labels, tokens)

        lengths = [len(sequence) for sequence in sequences_list]
        splitted_logits = torch.split(logits, lengths, dim=0)
        splitted_logits = [logits.numpy() for logits in splitted_logits]

        return splitted_logits

    def compute_probabilities(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        tokens_list: List[str] = None,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> List[Dict[int, Dict[str, float]]]:
        """Function that computes the probabilities over amino-acids from sequences.

        It takes as inputs a list of sequences and returns a list of dictionaries.
        Each dictionary contains the probabilities over the natural amino-acids for each
        position in the sequence. The keys represent the positions (indexed
        starting with 0) and the values are dictionaries of probabilities over
        the natural amino-acids for this position.

        In these dictionaries, the keys are the amino-acids and the value
        the corresponding probabilities.

        Args:
            sequences_list: List of sequences
            batch_size: number of sequences to consider for the forward pass
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            tokens_list: List of tokens to consider
            silent : display or not progress bar
        Returns:
            List[Dict[int, Dict[str, float]]]: dictionaries of probabilities per seq
        """
        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        _check_sequence(sequences_list, self.model_dir, 1024)
        _check_memory_logits(sequences_list, self.vocab_size, pass_mode)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        logits, _ = self._filter_logits(logits, labels, tokens)

        lengths = [len(sequence) for sequence in sequences_list]
        splitted_logits = torch.split(logits, lengths, dim=0)

        softmax = torch.nn.Softmax(dim=-1)
        splitted_probabilities = [softmax(logits) for logits in splitted_logits]

        def _get_probabilities_dict(probs: torch.Tensor) -> Dict[str, float]:
            return {
                aa: float(probs[i].cpu().numpy())
                for i, aa in enumerate(NATURAL_AAS_LIST)
            }

        probabilities = [
            {
                key: _get_probabilities_dict(value)
                for key, value in dict(enumerate(split)).items()
            }
            for split in splitted_probabilities
        ]

        return probabilities

    def compute_loglikelihood(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        tokens_list: List[str] = None,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> np.ndarray:
        """Function that computes loglikelihoods of sequences

        Args:
            sequences_list: List of sequences
            batch_size: Batch size
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            tokens_list: List of tokens to consider

        Returns:
            torch.Tensor: loglikelihoods in numpy format
        """
        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        _check_sequence(sequences_list, self.model_dir, 1024)
        _check_memory_logits(sequences_list, self.vocab_size, pass_mode)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        loglikelihoods = self._filter_loglikelihoods(logits, labels, tokens)

        return loglikelihoods.numpy()

    def compute_embeddings(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        pool_mode: Tuple[str, ...] = ("cls", "mean"),
        tokens_list: List[str] = None,
        silent: bool = False,
    ) -> Dict[str, np.ndarray]:
        """Function that computes embeddings of sequences.

        The embedding has a size (n_sequence, num_tokens, embeddings_size) so we use
        an aggregation function specified in pool_mode to aggregate the tensor on
        the num_tokens dimension. 'mean' signifies that we take the mean over the
        num_tokens dimension.

        Args:
            sequences_list: List of sequences
            batch_size: Batch size
            pool_mode: Mode of pooling ('cls', 'mean', 'min', 'max)
            tokens_list: List of tokens to consider
            silent : whereas to display or not progress bar
        Returns:
            torch.Tensor: Tensor of shape [number_of_sequences, embeddings_size]
        """
        if "full" in pool_mode and not all(
            len(s) == len(sequences_list[0]) for s in sequences_list
        ):
            raise Exception(
                'Sequences must be of same length when pool_mode = ("full",)'
            )

        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        _check_sequence(sequences_list, self.model_dir, 1024)
        _check_memory_embeddings(sequences_list, self.embeddings_size, pool_mode)

        inputs, _, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )
        embeddings_dict = dict(zip(pool_mode, [torch.Tensor()] * len(pool_mode)))

        for batch_inputs in tqdm(
            self._generate_chunks(inputs, batch_size),
            total=self._get_num_batch_iter(inputs, batch_size),
            disable=silent,
        ):
            _, batch_embeddings = self._model_pass(batch_inputs)
            batch_labels = batch_inputs["input_ids"]

            batch_embeddings_dict = self._filter_and_pool_embeddings(
                batch_embeddings, batch_labels, tokens, pool_mode
            )

            for key in pool_mode:
                embeddings_dict[key] = torch.cat(
                    (embeddings_dict[key], batch_embeddings_dict[key]), dim=0
                )

        return {key: value.numpy() for key, value in embeddings_dict.items()}

    def compute_accuracy(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        pass_mode: str = "forward",
        tokens_list: List[str] = None,
        silent: bool = False,
    ) -> float:
        """Compute model accuracy from the input sequences

        Args:
            sequences_list: [description]
            batch_size: [description]. Defaults to 1.
            pass_mode: [description]. Defaults to "forward".
            tokens_list: [description]. Defaults to None.
            silent: whereas to display or not progress bar
        Returns:
            [type]: [description]
        """
        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        _check_sequence(sequences_list, self.model_dir, 1024)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        logits, labels = self._filter_logits(logits, labels, tokens)
        accuracy = self._compute_accuracy(logits, labels)

        return accuracy

    def compute_calibration(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        pass_mode: str = "forward",
        tokens_list: Optional[List[str]] = None,
        n_bins: int = 10,
        silent: bool = False,
    ) -> Dict[str, Any]:
        """Compute model calibration from the input sequences

        Args:
            sequences_list : [description]
            batch_size : [description]. Defaults to 1.
            pass_mode : [description]. Defaults to "forward".
            tokens_list : [description]. Defaults to None.
            n_bins : [description]. Defaults to 10.
            silent: display or not progress bar
        Returns:
            [type]: [description]
        """
        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        _check_sequence(sequences_list, self.model_dir, 1024)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences_list, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        logits, labels = self._filter_logits(logits, labels, tokens)
        calibration_dict = self._compute_calibration(logits, labels, n_bins)

        return calibration_dict
