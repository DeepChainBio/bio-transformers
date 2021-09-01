"""
This script defines a generic template class for any language model.
Both ESM and Rostlab language models should implement this class.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch
from ray.actor import ActorHandle


class LanguageModel(ABC):
    """
    Class that implements a language model.
    """

    def __init__(self, model_dir: str, device):
        self._model_dir = model_dir
        self._device = device
        self.is_msa = False

    @property
    def model_id(self) -> str:
        """Model ID, as specified in the model directory"""
        return self._model_dir.lower()

    @property
    @abstractmethod
    def clean_model_id(self) -> str:
        """Clean model ID (in case the model directory is not)"""
        pass

    @property
    @abstractmethod
    def model_vocabulary(self) -> List[str]:
        """Returns the whole vocabulary list"""
        pass

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Returns the whole vocabulary size"""
        pass

    @property
    @abstractmethod
    def mask_token(self) -> str:
        """Representation of the mask token (as a string)"""
        pass

    @property
    @abstractmethod
    def pad_token(self) -> str:
        """Representation of the pad token (as a string)"""
        pass

    @property
    @abstractmethod
    def begin_token(self) -> str:
        """Representation of the beginning of sentence token (as a string)"""
        pass

    @property
    @abstractmethod
    def end_token(self) -> str:
        """Representation of the end of sentence token (as a string)."""
        pass

    @property
    @abstractmethod
    def does_end_token_exist(self) -> bool:
        """Returns true if a end of sequence token exists"""
        pass

    @property
    @abstractmethod
    def token_to_id(self):
        """Returns a function which maps tokens to IDs"""
        pass

    @property
    @abstractmethod
    def embeddings_size(self) -> int:
        """Returns size of the embeddings"""
        pass

    @abstractmethod
    def process_sequences_and_tokens(
        self,
        sequences_list: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Function to transform tokens string to IDs; it depends on the model used"""
        pass

    @property
    @abstractmethod
    def model(self) -> torch.nn.Module:
        """Return torch model."""
        pass

    @abstractmethod
    def _load_model(self, path: str):
        """Load model."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def get_alphabet_dataloader(self):
        """Define an alphabet mapping for common method between
        protbert and ESM
        """
        pass
