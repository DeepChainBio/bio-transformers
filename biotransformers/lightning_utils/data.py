import functools
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from esm.data import Alphabet, BatchConverter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

NATURAL_AAS_LIST = list("ACDEFGHIKLMNPQRSTVWY")


def mask_seq(
    seq: str,
    tokens: torch.Tensor,
    prepend_bos: bool,
    mask_idx: int,
    pad_idx: int,
    masking_ratio: float,
    masking_prob: float,
    random_token_prob: float,
    random_token_indices: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask one sequence randomly.

    Args:
        seq: string of the sequence.
        tokens: tokens corresponding to the sequence, length can be longer than the seq.
        prepend_bos: if tokenizer adds <bos> token
        mask_idx: index of the mask token
        pad_idx: index of the padding token
        masking_ratio: ratio of tokens to be masked.
        masking_prob: probability that the chose token is replaced with a mask token.
        random_token_prob: probability that the chose token is replaced with a random token.
        random_token_indices: list of token indices that random replacement selects from.

    Returns:
        tokens: masked tokens
        targets: same length as tokens
    """
    # init
    seq_len = len(seq)
    mask_num = int(np.ceil(seq_len * masking_ratio))
    targets = tokens.detach().clone()
    # sample indices
    mask_indices = sorted(
        np.random.choice(seq_len, mask_num, replace=False) + int(prepend_bos)
    )
    # mask tokens
    for idx in mask_indices:
        rand = np.random.random()

        # replace with mask
        if rand < masking_prob:
            tokens[idx] = mask_idx

        # replace with random token
        elif rand < masking_prob + random_token_prob:
            tokens[idx] = np.random.choice(random_token_indices, 1)[0]

    # generate targets
    non_mask_indices = [i for i in range(seq_len) if i not in mask_indices]
    targets[non_mask_indices] = pad_idx

    return tokens, targets


def worker_init_fn(worker_id: int):
    """Set numpy random seed for each worker.

    https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359

    Args:
        worker_id: unique id for each worker
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def collate_fn(
    samples: Sequence[Tuple[str, str]],
    tokenizer: BatchConverter,
    alphabet: Alphabet,
    masking_ratio: float,
    masking_prob: float,
    random_token_prob: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Collate function to mask tokens.

    Args:
        samples: a sequences of (label, seq).
        tokenizer: facebook tokenizer, that accepts sequences of (label, seq_str)
            and outputs (labels, seq_strs, tokens).
        alphabet: facebook alphabet.
        masking_ratio: ratio of tokens to be masked.
        masking_prob: probability that the chose token is replaced with a mask token.
        random_token_prob: probability that the chose token is replaced with a random token.

    Returns:
        tokens: model input
        targets: model target
        mask_indices: indices of masked tokens
    """
    random_token_indices = [alphabet.tok_to_idx[aa] for aa in NATURAL_AAS_LIST]
    _, seqs, tokens = tokenizer(samples)

    tokens_list, targets_list = [], []
    for i, seq in enumerate(seqs):
        tokens_i, targets_i = mask_seq(
            seq=seq,
            tokens=tokens[i, :],
            prepend_bos=alphabet.prepend_bos,
            mask_idx=alphabet.mask_idx,
            pad_idx=alphabet.padding_idx,
            masking_ratio=masking_ratio,
            masking_prob=masking_prob,
            random_token_prob=random_token_prob,
            random_token_indices=random_token_indices,
        )
        tokens_list.append(tokens_i)
        targets_list.append(targets_i)

    tokens = torch.stack(tokens_list)
    targets = torch.stack(targets_list)

    return tokens, targets


def create_dataset(
    sequences: List[str],
    alphabet: Alphabet,
    filter_len: bool,
    batch_size: int,
    masking_ratio: float,
    masking_prob: float,
    random_token_prob: float,
    num_workers: int = 0,
) -> DataLoader:
    """Create the PyTorch Dataset.

    Args:
        filenames: list of sequences
        alphabet: facebook alphabet.
        filter_len: whether filter data wrt len.
        batch_size: num samples per batchs
        num_workers: num of parallel data samplers
        masking_ratio: ratio of tokens to be masked.
        masking_prob: probability that the chose token is replaced with a mask token.
        random_token_prob: probability that the chose token is replaced with a random token.

    Returns:
        torch DataLoader
    """
    sequences = enumerate(sequences)  # type: ignore
    sequences = list(sequences)
    tokenizer = alphabet.get_batch_converter()

    loader = DataLoader(
        sequences,
        shuffle=True,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=functools.partial(
            collate_fn,
            tokenizer=tokenizer,
            alphabet=alphabet,
            masking_ratio=masking_ratio,
            masking_prob=masking_prob,
            random_token_prob=random_token_prob,
        ),
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init_fn,
    )
    return loader


class BioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_sequences: List[str],
        alphabet: Alphabet,
        filter_len: bool,
        batch_size: int,
        masking_ratio: float,
        masking_prob: float,
        random_token_prob: float,
        num_workers: int = 0,
        validation: bool = True,
    ):
        super().__init__()
        self.train_sequences = train_sequences
        self.alphabet = alphabet
        self.filter_len = filter_len
        self.batch_size = batch_size
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob
        self.num_workers = num_workers
        self.validation = validation

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.seq_train, self.seq_val = train_test_split(
                self.train_sequences, test_size=0.2
            )

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return create_dataset(
            sequences=self.seq_train,
            alphabet=self.alphabet,
            filter_len=self.filter_len,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            masking_ratio=self.masking_ratio,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
        )

    def val_dataloader(self):
        return create_dataset(
            sequences=self.seq_val,
            alphabet=self.alphabet,
            filter_len=self.filter_len,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            masking_ratio=self.masking_ratio,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
        )

    def test_dataloader(self):
        pass
