import functools
from collections import OrderedDict
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from biotransformers.utils.constant import NATURAL_AAS_LIST
from esm.data import Alphabet, BatchConverter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Sampler


class AlphabetDataLoader:
    """Class that carries tokenizer information"""

    def __init__(
        self,
        prepend_bos: bool,
        append_eos: bool,
        mask_idx: int,
        pad_idx: int,
        model_dir: str,
        lambda_toks_to_ids: Callable,
        lambda_tokenizer: Callable,
    ) -> None:
        self.prepend_bos = prepend_bos
        self.append_eos = append_eos
        self.mask_idx = mask_idx
        self.padding_idx = pad_idx
        self.model_dir = model_dir
        self.lambda_toks_to_ids = lambda_toks_to_ids
        self.lambda_tokenizer = lambda_tokenizer

    def tok_to_idx(self, x):
        return self.lambda_toks_to_ids(x)

    def tokenizer(self):
        """Return seq-token based on sequence"""
        return self.lambda_tokenizer


class CustomBatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    This custom BatchSampler is inspired from the torch class BatchSampler.
    It takes a list of indexes and shuffle the indexes at each epochs.

    Args:
        sampler (List): List of indexes. indexes are a collections of List[int],
        corresponding to the index of the protein sequence.
        batch_size (int): Size of mini-batch. 1 in our case, a batch are already of correct size.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not (type(batch_size) == int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got " "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        np.random.shuffle(self.sampler)
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class BatchDataset(Dataset):
    def __init__(self, sequences: List[str]) -> None:
        super().__init__()
        self.sequences = np.array(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index].tolist()


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
    mask_indices = sorted(np.random.choice(seq_len, mask_num, replace=False) + int(prepend_bos))
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
    random_token_indices = [alphabet.tok_to_idx(aa) for aa in NATURAL_AAS_LIST]
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


def _filter_sequence(sequences_list: List[str], model: str, filter_len: int) -> List[str]:
    """Function that filter the length of a sequence list

    Filtering depends on the type of model. It is automatically enforce as ESM1b
    does'nt manage sequence longer that 1024.

    Args:
        sequences_list : list of sequences
        model : name of the model
        length : length limit to consider
    Raises:
        ValueError is model filter_len < 0
    """

    if model == "esm1b_t33_650M_UR50S":
        filter_len = min(filter_len, 1024) if filter_len is not None else 1024
        return [seq for seq in sequences_list if len(seq) < filter_len]

    if filter_len is not None:
        if filter_len <= 0:
            raise ValueError("filter_len argument should be > 0")
        return [seq for seq in sequences_list if len(seq) < filter_len]

    return sequences_list


def get_batch_indices(
    sequence_strs,
    toks_per_batch: int,
    extra_toks_per_seq: int = 0,
) -> List[List[int]]:
    """Get the batch idx based on the number of tokens in sequences

    It computes a list of list of int which are the list of the indexes to consider
    to build a batch.
    Example:
        returning [[1,3,8],[4,7,10],[11],[12]] means that the first batch  will be
        composed of sequence at index 1,3,8 for the first batch, sequence 11 for the
        third batch. The idea is to consider a maximum number of tokens per batch.

    Args:
        sequence_strs: list of string
        filter_len :
        toks_per_batch (int): Maxi number of token per batch
        extra_toks_per_seq (int, optional): . Defaults to 0.

    Returns:
        List: List of batches indexes
    """
    buffer_type = List[int]
    sizes = [(len(s), i) for i, s in enumerate(sequence_strs)]
    sizes.sort()
    batches: List[buffer_type] = []
    buffer: buffer_type = []
    max_len = 0

    def _flush_current_buf():
        nonlocal max_len, buffer
        if len(buffer) == 0:
            return
        batches.append(buffer)
        buffer = []
        max_len = 0

    for sz, i in sizes:
        sz += extra_toks_per_seq
        if max(sz, max_len) * (len(buffer) + 1) > toks_per_batch:
            _flush_current_buf()
        max_len = max(max_len, sz)
        buffer.append(i)

    _flush_current_buf()
    return batches


def create_dataloader(
    sequences: List[str],
    alphabet: AlphabetDataLoader,
    filter_len: int,
    masking_ratio: float,
    masking_prob: float,
    random_token_prob: float,
    num_workers: int = 0,
    toks_per_batch: int = 128,
    extra_toks_per_seq: int = 2,
) -> DataLoader:
    """Create the PyTorch Dataset.

    Args:
        filenames: list of sequences
        alphabet: facebook alphabet.
        filter_len: whether filter data wrt len.batch_seq
        num_workers: num of parallel data samplers
        masking_ratio: ratio of tokens to be masked.
        masking_prob: probability that the chose token is replaced with a mask token.
        random_token_prob: probability that the chose token is replaced with a random token.

    Returns:
        torch DataLoader
    """
    sequences = _filter_sequence(sequences, alphabet.model_dir, filter_len)

    batches = get_batch_indices(
        sequences, toks_per_batch=toks_per_batch, extra_toks_per_seq=extra_toks_per_seq
    )

    dataset = BatchDataset(sequences)
    b_sampler = CustomBatchSampler(batches, batch_size=1, drop_last=False)

    loader = DataLoader(
        dataset,
        num_workers=num_workers,
        collate_fn=functools.partial(
            collate_fn,
            tokenizer=alphabet.tokenizer(),
            alphabet=alphabet,
            masking_ratio=masking_ratio,
            masking_prob=masking_prob,
            random_token_prob=random_token_prob,
        ),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        batch_sampler=b_sampler,
        sampler=None,
    )
    return loader


class BioDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_sequences: List[str],
        alphabet: AlphabetDataLoader,
        filter_len: int,
        masking_ratio: float,
        masking_prob: float,
        random_token_prob: float,
        toks_per_batch: int = 128,
        extra_toks_per_seq: int = 2,
        num_workers: int = 0,
        validation: bool = True,
    ):
        super().__init__()
        self.train_sequences = train_sequences
        self.alphabet = alphabet
        self.filter_len = filter_len
        self.masking_ratio = masking_ratio
        self.masking_prob = masking_prob
        self.random_token_prob = random_token_prob
        self.toks_per_batch = toks_per_batch
        self.extra_toks_per_seq = extra_toks_per_seq
        self.num_workers = num_workers
        self.validation = validation

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.seq_train, self.seq_val = train_test_split(self.train_sequences, test_size=0.2)

            # Optionally...
            # self.dims = tuple(self.mnist_train[0][0].shape)

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        # self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return create_dataloader(
            sequences=self.seq_train,
            alphabet=self.alphabet,
            filter_len=self.filter_len,
            num_workers=self.num_workers,
            masking_ratio=self.masking_ratio,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
            toks_per_batch=self.toks_per_batch,
            extra_toks_per_seq=self.extra_toks_per_seq,
        )

    def val_dataloader(self):
        return create_dataloader(
            sequences=self.seq_val,
            alphabet=self.alphabet,
            filter_len=self.filter_len,
            num_workers=self.num_workers,
            masking_ratio=self.masking_ratio,
            masking_prob=self.masking_prob,
            random_token_prob=self.random_token_prob,
            toks_per_batch=self.toks_per_batch,
            extra_toks_per_seq=self.extra_toks_per_seq,
        )

    def test_dataloader(self):
        pass
