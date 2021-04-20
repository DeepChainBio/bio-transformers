"""
This script defines a parent class for transformers, for which child classes which are
specific to a given transformers implementation can inherit.
It allows to derive probabilities, embeddings and log-likelihoods based on inputs
sequences, and displays some properties of the transformer model.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch

from .utils import (
    TransformersInferenceConfig,
    TransformersModelProperties,
    _check_sequence,
)

Probs = Dict[str, float]
SequenceProbs = List[Probs]
NATURAL_AAS = "ACDEFGHIKLMNPQRSTVWY"


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
    def model_vocab_ids(self) -> List[int]:
        """List of all vocabulary IDs to consider (as ints), which may be a subset
        of the model vocabulary (based on self.vocab_token_list)"""

    @property
    @abstractmethod
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""

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

    @staticmethod
    def softmaxbh(x: Union[np.array, torch.Tensor]) -> Union[np.array, torch.Tensor]:
        """
        Compute softmax values for each sets of scores in x. The max is computed
        on the last dimension.
        """
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / e_x.sum(axis=-1, keepdims=True)

    def update_oov_logits(self, logits: torch.tensor) -> torch.tensor:
        """Function to replace the logits by (-infinity) values if the tokens are not
           part of the selected vocabulary.

        Args:
            logits (torch.tensor): [description]

        Returns:
            torch.tensor: [description]
        """
        oov_vocab_ids = [
            id for id in range(logits.shape[-1]) if id not in self.model_vocab_ids
        ]
        logits[:, :, oov_vocab_ids] = -float("Inf")
        return logits

    @abstractmethod
    def _compute_forward_output(
        self, sequences_list: List[str], batch_size: int = 1
    ) -> Dict[torch.tensor, torch.tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration. The output is obtained
        by computing a forward pass through the model ("forward inference")

        Args:
            sequences_list (list): [description]
            batch_size (int):

        Returns:
            Dict[torch.tensor, torch.tensor]: [description]
        """
        return {"logits": NotImplemented, "embeddings": NotImplemented}

    @abstractmethod
    def _compute_masked_output(
        self,
        sequences_list: list,
        batch_size: int,
        inference_config: TransformersInferenceConfig,
    ) -> Dict[torch.tensor, torch.tensor]:
        """
        Function which computes logits and embeddings based on a list of sequences,
        a provided batch size and an inference configuration.

        The output is obtained by masking sequentially each amino-acid of the
        sequence (or only mutated amino-acids if specified this way), following a
        "masked inference" approach.

        Args:
            sequences_list (list): [description]
            batch_size (int): [description]

        Returns:
            Dict[torch.tensor, torch.tensor]: [description]
        """
        return {"logits": NotImplemented, "embeddings": NotImplemented}

    @staticmethod
    def extract_mutations_dict(orig_seq: str, full_sequences: List) -> List[Dict]:
        """
        Function which takes as input the original sequence, so as a list of
        sequences to analyse, and outputs a list of dictionaries with mutations ids as
        keys, and a tuple of original AA and mutated AA as values

        Args:
            orig_seq (str): [description]
            full_sequences (List): [description]

        Returns:
            List[Dict]: [description]
        """

        mutation_dicts_list = [
            {
                i: (orig_seq[i], seq[i])
                for i in range(len(orig_seq))
                if orig_seq[i] != seq[i]
            }
            for seq in full_sequences
        ]

        return mutation_dicts_list

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

    def _compute_probabilities_and_embeddings(
        self, sequences_list: List[str], batch_size: int
    ) -> Tuple[SequenceProbs, List[np.array]]:
        """
        For each position in each sequence returns the probabilities over
        the amino-acids for this position.


        Args:
            sequences_list (List[str]): [description]
            batch_size (int): [description]

        Returns:
            Tuple[SequenceProbs, List[np.array]]:
                probs_list : a list of size len(sequences_list), where each item is a list
                of length len(sequence) with the probabilities
                for each standard amino acid of the corresponding sequence
        """

        if self.mask_bool:
            output_dict = self._compute_masked_output(sequences_list, batch_size)
            logits, embeddings = output_dict["logits"], output_dict["embeddings"]

        else:
            output_dict = self._compute_forward_output(sequences_list, batch_size)
            logits, embeddings = output_dict["logits"], output_dict["embeddings"]

        # Turn into a list of probabilities - Remove CLS token from logits
        probs_list = [probs for probs in self.softmaxbh(logits[:, 1:].data.numpy())]

        # Define last position to consider for the embeddings
        # (remove padding but keep beginning and eventually end of sentence tokens)
        additional_pos = 1 if self.model_property.begin_token else 0
        additional_pos += 1 if self.model_property.end_token else 0
        if embeddings is not None:
            embeddings_list = [
                embedding_seq[: (seq_len + additional_pos)].data.numpy()
                for embedding_seq, seq_len in zip(
                    embeddings, [len(seq) for seq in sequences_list]
                )
            ]
        else:
            embeddings_list = None

        return probs_list, embeddings_list

    def compute_loglikelihood(
        self, sequences_list: List[str], mutation_dicts_list=None, batch_size: int = 1
    ) -> List[float]:
        """
        Computes the log likelihood of a sequence of amino-acids, either based on
        probabilities kept in memory, or based on a path to a pickle.

        Args:
            sequences_list (List[str]): [description]
            mutation_dicts_list ([type], optional): [description]. Defaults to None.
            batch_size (int, optional): [description]. Defaults to 1.

        Returns:
            List[float]: [description]
        """
        _check_sequence(sequences_list, self.model_dir, 1024)
        probs_list, _ = self._compute_probabilities_and_embeddings(
            sequences_list, batch_size=batch_size
        )

        if mutation_dicts_list is None:
            # Look at all amino-acids in each sequence
            scores = [
                [
                    probs[self.token_to_id(aa)]
                    for probs, aa in zip(probs_tensor, list(sequence))
                ]
                for sequence, probs_tensor in zip(sequences_list, probs_list)
            ]
        else:
            # Only look at the amino-acids which where mutated
            scores = [
                [
                    probs[self.token_to_id(aa)]
                    for probs, aa in zip(
                        probs_tensor,
                        [
                            sequence[i]
                            for i in range(len(sequence))
                            if i in mutation_dicts_list[seq_id].keys()
                        ],
                    )
                ]
                for (seq_id, (sequence, probs_tensor)) in enumerate(
                    zip(sequences_list, probs_list)
                )
            ]
        log_scores = [np.log(np.asarray(score)) for score in scores]
        log_likelihood = [float(np.sum(log_score)) for log_score in log_scores]

        return log_likelihood

    def _pool_sequence_embeddings(
        self,
        embeddings_matrix: np.array,
        pooling_list: List[str],
        return_sequence=False,
    ) -> Dict:
        """
        Function which pools the embeddings of a sequence (all of its position) based
        on specified pooling approaches.

        Args:
            embeddings_matrix (np.array): [description]
            pooling_list (tuple): [description]

        Returns:
            Dict: Dictionnary with all specific pooling function as key,
                  and embedding vector as value. Always return BEGIN token
                  by default.
        """
        pool_dict = dict()

        pool_dict["cls"] = embeddings_matrix[0, :]
        embeddings_matrix = embeddings_matrix[1:, :]

        if ("mean" in pooling_list) | ("average" in pooling_list):
            pool_dict["mean"] = np.mean(embeddings_matrix, axis=0)

        if ("min" in pooling_list) | ("minimum" in pooling_list):
            pool_dict["min"] = np.min(embeddings_matrix, axis=0)

        if ("max" in pooling_list) | ("maximum" in pooling_list):
            pool_dict["max"] = np.max(embeddings_matrix, axis=0)

        if return_sequence:
            pool_dict["sequence"] = embeddings_matrix

        return pool_dict

    def compute_embeddings(
        self,
        sequences_list: List[str],
        batch_size: int = 1,
        pooling_list: Optional[List] = None,
        return_sequence: bool = False,
    ) -> Dict[str, np.array]:
        """
        Compute of embeddings for a list of sequence.
        Can provide multiple pooling function to aggregate the embedding
        The result is a dictionnary with a key for each pool function provided.

        Args:
            sequences_list (List[str]): [description]
            batch_size (int, optional): [description]. Defaults to 1.
            pooling_tuple (tuple, optional): [description]. Defaults to [].


        Returns:
            List[Dict]: The embeddings dictionnary is composed of the <CLS>  embedding
                         and with the pool function provided (ie: mean, max, min)
        """
        _check_sequence(sequences_list, self.model_dir, 1024)

        if pooling_list is None:
            pooling_list = []
        pooling_dict = dict()

        _, embedding_list = self._compute_probabilities_and_embeddings(
            sequences_list, batch_size=batch_size
        )

        pool_list = [
            self._pool_sequence_embeddings(embedding, pooling_list, return_sequence)
            for embedding in embedding_list
        ]

        cls_array = np.array([emb["cls"] for emb in pool_list])
        pooling_dict["cls"] = cls_array

        if len(pooling_list):
            pooling_dict.update(
                {
                    pool_key: np.array([emb[pool_key] for emb in pool_list])
                    for pool_key in pooling_list
                }
            )
        if return_sequence:
            pooling_dict.update({"sequence": [emb["sequence"] for emb in pool_list]})

        return pooling_dict
