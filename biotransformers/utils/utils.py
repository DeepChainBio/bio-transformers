import math
import os
from dataclasses import dataclass
from typing import List, Tuple

from Bio import SeqIO
from biotransformers.utils.logger import logger

log = logger("utils")


def convert_bytes_size(size_bytes: int) -> Tuple[str, bool]:
    """[summary]

    Args:
        size_bytes: size in bytes

    Returns:
        Tuple[str,bool]: return the size with correct units and a condition
        to display the warning message.
    """
    if size_bytes == 0:
        return "0B", False
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = int(round(size_bytes / p, 2))
    is_warning = i >= 3  # warning on size only for model in GB

    return "%s%s" % (s, size_name[i]), is_warning


def _check_memory_embeddings(
    sequences_list: List[str], embeddings_size: int, pool_mode: Tuple[str, ...]
):
    """Function to compute the memory taken by the embeddings with float64 number.

    Args:
        sequences_list: sequences of proteins
        embeddings_size : size of the embeddings vector, depends on the model
        pool_mode : aggregation function
    """
    num_of_sequences = len(sequences_list)
    emb_dict_len = len(pool_mode)
    tensor_memory_bits = 64  # double/float64
    memory_bits = num_of_sequences * embeddings_size * emb_dict_len * tensor_memory_bits
    memory_bytes = int(memory_bits / 8)
    memory_convert_bytes, is_warning = convert_bytes_size(memory_bytes)

    if is_warning:
        log.warning(
            "Embeddings will need about %s of memory."
            "Please make sure you have enough space",
            memory_convert_bytes,
        )


def _check_memory_logits(sequences_list: List[str], vocab_size: int, pass_mode: str):
    """Function to compute the memory taken by the logits with float64 number.

    Args:
        sequences_list ([type]): [description]
        vocab_size ([type]): [description]
        pass_mode ([type]): [description]
    """
    num_of_sequences = len(sequences_list)
    sum_seq_len = sum([len(seq) for seq in sequences_list])
    max_seq_len = max([len(seq) for seq in sequences_list])
    tensor_memory_bits = 64  # double/float64
    if pass_mode == "masked":
        memory_bits = sum_seq_len * max_seq_len * vocab_size * tensor_memory_bits
    elif pass_mode == "forward":
        memory_bits = num_of_sequences * max_seq_len * vocab_size * tensor_memory_bits

    memory_bytes = int(memory_bits / 8)
    memory_convert_bytes, is_warning = convert_bytes_size(memory_bytes)

    if is_warning:
        log.warning(
            "%s mode will need about %s of memory. Please make sure you have enough space",
            pass_mode,
            memory_convert_bytes,
        )


def _check_sequence(sequences_list: List[str], model: str, length: int):
    """Function that control sequence length

    Args:
        model : name of the model
        length : length limit to consider
    Raises:
        ValueError is model esm1b_t33_650M_UR50S and sequence_length >1024
    """

    if model == "esm1b_t33_650M_UR50S":
        is_longer = list(map(lambda x: len(x) > length, sequences_list))

        if sum(is_longer) > 0:
            raise ValueError(
                f"You cant't pass sequence with length more than {length} "
                f"with esm1b_t33_650M_UR50S, use esm1_t34_670M_UR100 or "
                f"filter the sequence length"
            )
    return


def load_fasta(path_fasta: str) -> List[str]:
    """Read and parse records from a fasta file

    Args:
        path_fasta: path of the fasta file

    Returns:
        List: List of sequences
    """
    return [str(record.seq) for record in SeqIO.parse(path_fasta, format="fasta")]


def get_logs_version(path_logs):
    """Get last version of logs folder to save model inside

    Args:
        path_logs (str): path of the logs/experiments folder
    """
    version = str(max([int(fold.split("_")[1]) for fold in os.listdir(path_logs)]))
    return "version_" + version


def format_backend(backend_list: List[str]) -> List[str]:
    """format of list to display"""
    return ["  *" + " " * 3 + model for model in backend_list]


@dataclass
class TransformersModelProperties:
    """Class to describe some model properties"""

    num_sep_tokens: int  # Number of separation tokens (beginning+end)
    begin_token: bool  # Whether there is a beginning of sentence token
    end_token: bool  # Whether there is an end of sentence token


@dataclass
class TransformersInferenceConfig:
    """
    Class to describe the inference configuration.
    - mask_bool: whether to compute a masked inference (masked_bool=True), or a forward
    ingerence (mask_bool=False)

    - mutation_dicts_list: list of dictionnary, which indicates, for each sequence, the
    position of the amino-acids which were mutated. This allows to compute "local"
     embeddings and probabilities only (on mutated amino-acids only).
     eg [{1: ('A', 'C')}, {1: ('A', 'W'), '4': ('W', 'C')}] means that in the first
     sequence, the position 1 in the original amino-acid was an 'A', now replaced by a
     'C', and in the second sequence two mutations happened, at the positions 1 and 4.
     See the extract_mutations_dict method of TransformersWrapper.

    - all_masks_forward_local_bool: whether to mask all amino-acids when a local forward
    approach is used.
    """

    mask_bool: bool
    mutation_dicts_list: List[dict] = None
    all_masks_forward_local_bool: bool = False
