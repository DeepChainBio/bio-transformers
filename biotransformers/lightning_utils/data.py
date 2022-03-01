import functools
import math
import random
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as dist
from esm.data import BatchConverter
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, Sampler


class AlphabetDataLoader:
    """Class that carries tokenizer information"""

    def __init__(
        self,
        prepend_bos: bool,
        append_eos: bool,
        mask_idx: int,
        pad_idx: int,
        standard_toks: List[str],
        model_dir: str,
        lambda_toks_to_ids: Callable,
        lambda_tokenizer: Callable,
    ) -> None:
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.mask_idx = mask_idx
        self.padding_idx = pad_idx
        self.standard_toks = standard_toks
        self.model_dir = model_dir
        self.lambda_toks_to_ids = lambda_toks_to_ids
        self.lambda_tokenizer = lambda_tokenizer

    def tok_to_idx(self, x):
        return self.lambda_toks_to_ids(x)

    def tokenizer(self):
        """Return seq-token based on sequence"""
        return self.lambda_tokenizer


def convert_ckpt_to_statedict(checkpoint_state_dict: OrderedDict) -> OrderedDict:
    """This function convert a state_dict coming form pytorch lightning checkpoint to
    a state_dict model that can be load directly in the bio-transformers model.

    The keys are updated so that it  m.jionatches those in the bio-transformers

    Args:
        checkpoint_state_dict: a state_dict loaded from a checkpoint
    """
    new_state_dict = OrderedDict()
    for k, v in checkpoint_state_dict.items():
        new_k = ".".join(k.split(".")[1:])  # remove model. prefix in key
        new_state_dict[new_k] = v.to("cpu")  # move tensor to cpu

    return new_state_dict


def worker_init_fn(worker_id: int):
    """Set numpy random seed for each worker.

    https://github.com/pytorch/pytorch/issues/5059#issuecomment-404232359

    Args:
        worker_id: unique id for each worker
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)


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


def collate_fn(
    samples: Sequence[Tuple[str, str]],
    tokenizer: BatchConverter,
    alphabet: AlphabetDataLoader,
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
    random_token_indices = [alphabet.tok_to_idx(aa) for aa in alphabet.standard_toks]
    seqs, tokens = tokenizer(
        samples[0]
    )  # take samples[0] because batch_sampler return list of list
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


def crop_sequence(sequence: str, crop_length: int) -> str:
    """If the length of the sequence is superior to crop_length, crop randomly
    the sequence to get the proper length."""
    if len(sequence) <= crop_length:
        return sequence
    else:
        start_idx = random.randint(0, len(sequence) - crop_length)
        return sequence[start_idx : (start_idx + crop_length)]


def get_batch_indices(
    sequence_strs: List[str],
    toks_per_batch: int,
    crop_sizes: Tuple[int, int] = (600, 1200),
    seed: int = 0,
) -> List[List[List[Tuple[int, int]]]]:
    """
    This sampler aims to create batches that do not contain fixed number of sequences
    but rather constant number of tokens. Some the batch can contain a few long
    sequences or multiple small ones.

    This sampler returns batches of indices to achieve this property. It also decides
    if sequences must be cropped and return the desired length. The cropping length is
    sampled randomly for each sequence at each epoch in the range of crop_sizes values.

    This sampler computes a list of list of tuple which contains indices and
    lengths of sequences inside the batch.

    Example:
        returning [[(1, 100), (3, 600)],[(4, 100), (7, 1200), (10, 600)], [(12, 1000)]]
        means that the first batch will be composed of sequence at index 1 and 3 with
        lengths 100 and 600. The third batch contains only sequence 12 with a length
        of 1000.

    Args:
        sequence_strs: list of string
        toks_per_batch (int): Maximum number of token per batch
        extra_toks_per_seq (int, optional): . Defaults to 0.
        crop_sizes (Tuple[int, int]): min and max sequence lengths when cropping
        seed (int): seed to be used for random generator

    Returns:
        List: List of batches indexes and lengths
    """
    min_size, max_size = crop_sizes
    buffer_type = List[Tuple[int, int]]

    rand_generator = random.Random(seed)

    def crop_length(length: int, random_generator: random.Random) -> int:
        crop_size = random_generator.randint(min_size, max_size) - 2
        if length > crop_size:
            return crop_size
        else:
            return length

    sizes = [
        (crop_length(len(s), rand_generator), i) for i, s in enumerate(sequence_strs)
    ]
    min_length, max_length = min([t[0] for t in sizes]), max([t[0] for t in sizes])

    # if there is a large gap between min and max size, sort the list
    if min_length < 0.8 * max_length:
        sizes.sort()
    # otherwise shuffle it
    else:
        rand_generator.shuffle(sizes)

    batches: List[List[buffer_type]] = []
    buffer: buffer_type = []

    def _flush_current_buf():
        nonlocal buffer
        if len(buffer) == 0:
            return
        batches.append([buffer])
        buffer = []

    for seq_length, i in sizes:

        num_toks_if_seq_is_added = (len(buffer) + 1) * max(
            [seq_length] + [b[1] for b in buffer]
        )

        if num_toks_if_seq_is_added > toks_per_batch:
            _flush_current_buf()

        buffer.append((i, seq_length))

    _flush_current_buf()

    rand_generator.shuffle(batches)
    return batches


class BatchWithConstantNumberTokensSampler(Sampler):
    """
    Sampler that returns batches of sequences indices in the dataset so that to ensure
    not a fixed number of sequences per batch but rather a fixed number of tokens per
    batch. This sampler also takes into account that we may want to crop dynamically
    sequences when sampling and thus returns in addition to indices, desired cropping
    lengths to inform the dataloader.
    """

    def __init__(
        self,
        sequence_strs: List[str],
        toks_per_batch: int,
        crop_sizes: Tuple[int, int] = (512, 1024),
    ):
        Sampler.__init__(self, data_source=None)
        self._sequence_strs = sequence_strs
        self._toks_per_batch = toks_per_batch
        self._crop_sizes = crop_sizes
        self._init_batches = get_batch_indices(
            sequence_strs=sequence_strs,
            toks_per_batch=toks_per_batch,
            crop_sizes=crop_sizes,
        )

    def __len__(self):
        return len(self._init_batches)

    def __iter__(self):
        yield from get_batch_indices(
            sequence_strs=self._sequence_strs,
            toks_per_batch=self._toks_per_batch,
            crop_sizes=self._crop_sizes,
        )


class DistributedBatchWithConstantNumberTokensSampler(Sampler):
    """
    Sampler that returns batches of sequences indices in the dataset so that to ensure
    not a fixed number of sequences per batch but rather a fixed number of tokens per
    batch. This sampler also takes into account that we may want to crop dynamically
    sequences when sampling and thus returns in addition to indices, desired cropping
    lengths to inform the dataloader. This version of the sampler is distributed to
    be used with DDP accelerator.
    """

    def __init__(
        self,
        sequence_strs: List[str],
        toks_per_batch: int,
        crop_sizes: Tuple[int, int] = (512, 1024),
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
    ):
        Sampler.__init__(self, data_source=None)

        # Replicate Torch Distributed Sampler logic
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1)
            )

        self._num_replicas = num_replicas
        self._rank = rank
        self._epoch = 0
        self._seed = seed

        self._sequence_strs = sequence_strs
        self._toks_per_batch = toks_per_batch
        self._crop_sizes = crop_sizes
        self._init_batches = get_batch_indices(
            sequence_strs=sequence_strs,
            toks_per_batch=toks_per_batch,
            crop_sizes=crop_sizes,
            seed=self._seed + self._epoch,
        )
        self._num_samples = math.ceil(len(self._init_batches) / self._num_replicas)
        self._total_size = self._num_samples * self._num_replicas

    def __len__(self) -> int:
        return self._num_samples

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __iter__(self):

        # generate batches with constant number of tokens
        batches = get_batch_indices(
            sequence_strs=self._sequence_strs,
            toks_per_batch=self._toks_per_batch,
            crop_sizes=self._crop_sizes,
            seed=self._seed + self._epoch,
        )

        # shuffle the indices
        rng = np.random.default_rng(seed=self._seed + self._epoch)
        indices = list(rng.permutation(len(batches)))

        # add extra samples to make it evenly divisible
        padding_size = self._total_size - len(indices)
        if padding_size <= len(indices):
            indices += indices[:padding_size]
        else:
            indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        assert len(indices) == self._total_size

        # subsample (to get batches for this worker)
        indices = indices[self._rank : self._total_size : self._num_replicas]
        assert len(indices) == self._num_samples

        # get corresponding batches
        batches = [batches[i] for i in indices]

        # return iterator
        yield from batches


class BatchWithConstantNumberTokensDataset(Dataset):
    """
    Dataset class to work in pair with the BatchWithConstantNumberTokensSampler.
    """

    def __init__(self, sequences: List[str]):
        super().__init__()
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, sampler_out) -> List[str]:
        indices, lengths = zip(*sampler_out)
        sequences = [
            crop_sequence(self.sequences[i], length)
            for i, length in zip(indices, lengths)
        ]
        return sequences


class BatchWithConstantNumberTokensDataModule(LightningDataModule):
    def __init__(
        self,
        train_sequences: List[str],
        validation_sequences: List[str],
        alphabet: AlphabetDataLoader,
        masking_ratio: float,
        masking_prob: float,
        random_token_prob: float,
        num_workers: int,
        toks_per_batch: int,
        crop_sizes: Tuple[int, int] = (512, 1024),
    ):
        LightningDataModule.__init__(self)
        self._train_sequences = train_sequences
        self._validation_sequences = validation_sequences
        self._alphabet = alphabet
        self._masking_ratio = masking_ratio
        self._masking_prob = masking_prob
        self._random_token_prob = random_token_prob
        self._num_workers = num_workers
        self._toks_per_batch = toks_per_batch
        self._crop_sizes = crop_sizes

    def _get_dataloader(self, sequences: List[str]) -> DataLoader:
        dataset = BatchWithConstantNumberTokensDataset(sequences)
        batch_sampler = DistributedBatchWithConstantNumberTokensSampler(
            sequence_strs=sequences,
            toks_per_batch=self._toks_per_batch,
            crop_sizes=self._crop_sizes,
        )

        loader = DataLoader(
            dataset,
            num_workers=self._num_workers,
            collate_fn=functools.partial(
                collate_fn,
                tokenizer=self._alphabet.tokenizer(),
                alphabet=self._alphabet,
                masking_ratio=self._masking_ratio,
                masking_prob=self._masking_prob,
                random_token_prob=self._random_token_prob,
            ),
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            batch_sampler=batch_sampler,
            sampler=None,
        )
        return loader

    def train_dataloader(self):
        return self._get_dataloader(self._train_sequences)

    def val_dataloader(self):
        return self._get_dataloader(self._validation_sequences)
