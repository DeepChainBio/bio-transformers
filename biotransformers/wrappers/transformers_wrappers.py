"""
This script defines a parent class for transformers, for which child classes which are
specific to a given transformers implementation can inherit.
It allows to derive probabilities, embeddings and log-likelihoods based on inputs
sequences, and displays some properties of the transformer model.
"""
import os
from abc import ABC, abstractmethod
from os.path import join
from typing import Any, Dict, Generator, Iterable, List, Optional, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.tensor
from biotransformers.lightning_utils.data import (
    AlphabetDataLoader,
    convert_ckpt_to_statedict,
)
from biotransformers.utils.constant import NATURAL_AAS_LIST
from biotransformers.utils.gpus_utils import set_device
from biotransformers.utils.logger import logger  # noqa
from biotransformers.utils.utils import (
    _check_memory_embeddings,
    _check_memory_logits,
    _check_sequence,
    get_logs_version,
    load_fasta,
)

# from esm.data import read_fasta
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.nn import DataParallel
from torch.nn import functional as F  # noqa: N812
from tqdm import tqdm

from ..lightning_utils.data import BioDataModule
from ..lightning_utils.models import LightningModule

log = logger("transformers_wrapper")


class TransformersWrapper(ABC):
    """
    Abstract class that uses pretrained transformers model to evaluate
    a protein likelihood so as other insights.
    """

    def __init__(
        self,
        model_dir: str,
        _device: Optional[str] = None,
        multi_gpu: bool = False,
        mask_bool: bool = False,
    ):
        """Initialize Transformers wrapper

        Args:
            model_dir: name directory of the pretrained model
            _device: type of device to use (cpu or cuda).
            multi_gpu: turn on to True to use multigpu
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
    ) -> Tuple[Dict[str, torch.Tensor], torch.tensor, List[int]]:
        """Function to transform tokens string to IDs; it depends on the model used"""

    @abstractmethod
    def _model_pass(
        self, model_inputs: Dict[str, torch.tensor]
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Function which computes logits and embeddings based on a list of sequences"""

    @abstractmethod
    def _get_alphabet_dataloader(self) -> AlphabetDataLoader:
        """Function to build custom alphanbet"""

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
        """Compute logits and embeddings

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
        sequences: Union[List[str], str],
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

        if isinstance(sequences, str):
            sequences = load_fasta(sequences)

        _check_sequence(sequences, self.model_dir, 1024)
        _check_memory_logits(sequences, self.vocab_size, pass_mode)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        logits, labels = self._filter_logits(logits, labels, tokens)

        lengths = [len(sequence) for sequence in sequences]
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
        sequences: Union[List[str], str],
        batch_size: int = 1,
        tokens_list: List[str] = None,
        pass_mode: str = "forward",
        silent: bool = False,
    ) -> np.ndarray:
        """Function that computes loglikelihoods of sequences

        Args:
            sequences: List of sequences
            batch_size: Batch size
            pass_mode: Mode of model evaluation ('forward' or 'masked')
            tokens_list: List of tokens to consider

        Returns:
            torch.Tensor: loglikelihoods in numpy format
        """
        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        if isinstance(sequences, str):
            sequences = load_fasta(sequences)

        _check_sequence(sequences, self.model_dir, 1024)
        _check_memory_logits(sequences, self.vocab_size, pass_mode)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        loglikelihoods = self._filter_loglikelihoods(logits, labels, tokens)

        return loglikelihoods.numpy()

    def compute_embeddings(
        self,
        sequences: Union[List[str], str],
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
            sequences: List of sequences or path of fasta file
            batch_size: Batch size
            pool_mode: Mode of pooling ('cls', 'mean', 'min', 'max)
            tokens_list: List of tokens to consider
            silent : whereas to display or not progress bar
        Returns:
            torch.Tensor: Tensor of shape [number_of_sequences, embeddings_size]
        """
        if "full" in pool_mode and not all(
            len(s) == len(sequences[0]) for s in sequences
        ):
            raise Exception(
                'Sequences must be of same length when pool_mode = ("full",)'
            )

        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        if isinstance(sequences, str):
            sequences = load_fasta(sequences)

        _check_sequence(sequences, self.model_dir, 1024)
        _check_memory_embeddings(sequences, self.embeddings_size, pool_mode)

        inputs, _, tokens = self._process_sequences_and_tokens(sequences, tokens_list)
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
        sequences: Union[List[str], str],
        batch_size: int = 1,
        pass_mode: str = "forward",
        tokens_list: List[str] = None,
        silent: bool = False,
    ) -> float:
        """Compute model accuracy from the input sequences

        Args:
            sequences (Union[List[str],str]): list of sequence or fasta file
            batch_size ([type], optional): [description]. Defaults to 1.
            pass_mode ([type], optional): [description]. Defaults to "forward".
            tokens_list ([type], optional): [description]. Defaults to None.

        Returns:
            [type]: [description]
        """
        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        if isinstance(sequences, str):
            sequences = load_fasta(sequences)
        _check_sequence(sequences, self.model_dir, 1024)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        logits, labels = self._filter_logits(logits, labels, tokens)
        accuracy = self._compute_accuracy(logits, labels)

        return accuracy

    def compute_calibration(
        self,
        sequences: Union[List[str], str],
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
            [Dict]: [description]
        """
        if tokens_list is None:
            tokens_list = NATURAL_AAS_LIST

        if isinstance(sequences, str):
            sequences = load_fasta(sequences)

        _check_sequence(sequences, self.model_dir, 1024)

        inputs, labels, tokens = self._process_sequences_and_tokens(
            sequences, tokens_list
        )
        logits = self._compute_logits(inputs, batch_size, pass_mode, silent=silent)
        logits, labels = self._filter_logits(logits, labels, tokens)
        calibration_dict = self._compute_calibration(logits, labels, n_bins)

        return calibration_dict

    def load_model(self, model_dir: str, map_location=None):
        """Load state_dict a finetune pytorch model ro a checkpoint directory

        More informations about how to load a model with map_location:
            https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-model-for-inference

        Args:
            model_dir: path file of the pt model or checkpoint.
                       the checkpoint should be a pytorch model checkpoint
        """
        if not os.path.isfile(model_dir):
            raise FileNotFoundError

        if model_dir.endswith(".pt"):
            load_model = torch.load(model_dir)
            log.info("Load model %s" % model_dir)
        elif model_dir.endswith(".ckpt"):
            load_model = convert_ckpt_to_statedict(torch.load(model_dir)["state_dict"])
            log.info("Load checkpoint %s" % model_dir)
        else:
            raise ValueError("Expecting a .pt or .ckpt file")

        if self.multi_gpu:
            self.model.module.load_state_dict(load_model, map_location)  # type: ignore
        else:
            self.model.load_state_dict(load_model, map_location)  # type: ignore
            self.model.eval()  # type: ignore

    def save_model(self, exp_path: str, lightning_model: pl.LightningModule) -> str:
        """Save pytorch model in logs directory

        Args:
            exp_path (str): path of the experiments directory in the logs
        """
        version = get_logs_version(exp_path)
        model_dir = self.model_dir.replace("/", "_")
        save_name = os.path.join(exp_path, version, model_dir + "_finetuned.pt")
        torch.save(lightning_model.model.state_dict(), save_name)

        return save_name

    def train_masked(
        self,
        train_sequences: Union[List[str], str],
        lr: float = 1.0e-5,
        warmup_updates: int = 1024,
        warmup_init_lr: float = 1e-7,
        epochs: int = 10,
        batch_size: int = 2,
        acc_batch_size: int = 256,
        masking_ratio: float = 0.025,
        masking_prob: float = 0.8,
        random_token_prob: float = 0.15,
        toks_per_batch: int = 2048,
        filter_len=1024,
        accelerator: str = "ddp",
        amp_level: str = "O2",
        precision: int = 16,
        logs_save_dir: str = "logs",
        logs_name_exp: str = "finetune_masked",
        checkpoint: Optional[str] = None,
        save_last_checkpoint: bool = True,
    ):
        """Function to finetune a model on a specific dataset

        This function will finetune the choosen model on a dataset of
        sequences with pytorch ligthening. You can modify the masking ratio of AA
        in the arguments for better convergence.
        Be careful with the accelerator that you use. DDP accelerator will
        launch multiple python process and do not be use in a notebook.

        More informations on GPU/accelerator compatibility here :
            https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html
        The wisest choice would be to use DDP for multi-gpu training.

        Args:
            train_sequences : Could be a list of sequences or the path of a
                              fasta file with multiple seqRecords
            lr : learning rate for training phase. Defaults to 1.0e-5.
            warmup_updates : Number of warming updates, number of step while increasing
            the leraning rate. Defaults to 1024.
            warmup_init_lr :  Initial lr for warming_update. Defaults to 1e-7.
            epochs :  number of epoch for training. Defaults to 10.
            batch_size :  number of sequence to consider in a batch. Defaults to 2.
            acc_batch_size : accumulated batch size Defaults to 2048.
            masking_ratio : ratio of tokens to be masked. Defaults to 0.025.
            masking_prob :  probability that the chose token is replaced with a mask token.
                            Defaults to 0.8.
            random_token_prob : probability that the chose token is replaced with a random token.
                                Defaults to 0.1.
            toks_per_batch: Maximum number of token to consider in a batch.Defaults to 2048.
                            This argument will set the number of sequences in a batch, which
                            is dynamically computed. Batch size use accumulate_grad_batches to compute
                            accumulate_grad_batches parameter.
            extra_toks_per_seq: Defaults to 2,
            filter_len : Size of sequence to filter. Defaults to 1024. (NOT USED)
            accelerator: type of accelerator for mutli-gpu processing (DPP recommanded)
            amp_level: allow mixed precision. Defaults to '02'
            precision: reducing precision allows to decrease the GPU memory needed.
                       Defaults to 16 (float16)
            logs_save_dir : Defaults directory to logs.
            logs_name_exp: Name of the experience in the logs.
            checkpoint : Path to a checkpoint file to restore training session.
            save_last_checkpoint: Save last checkpoint and 2 best trainings models
                                  to restore training session. Take a large amout of time and memory.
        """
        if isinstance(train_sequences, str):
            train_sequences = load_fasta(train_sequences)
        _check_sequence(train_sequences, self.model_dir, 1024)  # noqa: ignore

        fit_model = self.model.module if self.multi_gpu else self.model  # type: ignore
        alphabet = self._get_alphabet_dataloader()

        extra_toks_per_seq = int(alphabet.prepend_bos) + int(alphabet.append_eos)
        lightning_model = LightningModule(
            model=fit_model,
            alphabet=alphabet,
            lr=lr,
            warmup_updates=warmup_updates,
            warmup_init_lr=warmup_init_lr,
            warmup_end_lr=lr,
        )

        data_module = BioDataModule(
            train_sequences,
            alphabet,
            filter_len,
            batch_size,
            masking_ratio,
            masking_prob,
            random_token_prob,
            toks_per_batch,
            extra_toks_per_seq,
        )

        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
        else:
            log.warning("You try to train a transformers without GPU.")
            return

        logger = CSVLogger(logs_save_dir, name=logs_name_exp)
        checkpoint_callback = None

        if save_last_checkpoint:
            checkpoint_callback = [
                ModelCheckpoint(
                    save_last=True,
                    save_top_k=2,
                    mode="max",
                    monitor="val_acc",
                    every_n_val_epochs=3,
                )
            ]

        trainer = Trainer(
            gpus=n_gpus,
            amp_level=amp_level,
            precision=precision,
            accumulate_grad_batches=acc_batch_size // batch_size,
            max_epochs=epochs,
            logger=logger,
            accelerator=accelerator,
            replace_sampler_ddp=False,
            resume_from_checkpoint=checkpoint,
            callbacks=checkpoint_callback,
        )

        trainer.fit(lightning_model, data_module)

        save_path = join(logs_save_dir, logs_name_exp)
        if accelerator == "ddp":
            rank = os.environ.get("LOCAL_RANK", None)
            rank = int(rank) if rank is not None else None  # type: ignore
            if rank == 0:
                save_name = self.save_model(save_path, lightning_model)
                log.info("Model save at %s." % save_name)
        else:
            save_name = self.save_model(save_path, lightning_model)
            log.info("Model save at %s." % save_name)

        if self.multi_gpu:
            self.model = DataParallel(lightning_model.model).to(self._device)
        else:
            self.model = lightning_model.model.to(self._device)

        log.info("Training completed.")
