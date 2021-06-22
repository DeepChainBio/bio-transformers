import math
import os
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, List, Tuple, Union

import numpy as np
from Bio import SeqIO
from biotransformers.utils.constant import BACKEND_LIST
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
            "Embeddings will need about %s of memory." "Please make sure you have enough space",
            memory_convert_bytes,
        )


def _check_memory_logits(sequences_list: List[str], vocab_size: int, pass_mode: str):
    """Function to compute the memory taken by the logits with float64 number.

    Args:
        sequences_list (str): sequences of proteins
        vocab_size (int]): Size of the vocabulary
        pass_mode (str): 'forward' or 'masked'
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


def _check_tokens_list(sequences_list: List[str], tokens_list: List[str]):
    """Function that check if the list of tokens contains at least the tokens
    that are in the sequences.

    Args:
        sequences_list : list of sequences
        tokens_list : list of tokens to consider
    Raises:
        ValueError if some tokens in the sequences are not in the tokens_list
    """
    tokens = []
    for sequence in sequences_list:
        tokens += list(sequence)
        tokens = list(set(tokens))
    for token in tokens:
        if token not in tokens_list:
            raise ValueError(
                f"Token {token} is present in the sequences but not in the tokens_list."
            )


def _check_batch_size(batch_size: int, num_gpus: int):
    if not isinstance(batch_size, int):
        raise TypeError("batch_size should be of type int")
    if num_gpus > 1:
        if batch_size < num_gpus:
            raise ValueError("With num_gpus>1, batch_size should be at least equal to num_gpus.")


def _get_num_batch_iter(model_inputs: Dict[str, Any], batch_size: int) -> int:
    """
    Get the number of batches when spliting model_inputs into chunks
    of size batch_size.
    """
    num_of_sequences = model_inputs["input_ids"].shape[0]
    num_batch_iter = int(np.ceil(num_of_sequences / batch_size))
    return num_batch_iter


def _generate_chunks(
    model_inputs: Dict[str, Any], batch_size: int
) -> Generator[Dict[str, Iterable], None, None]:
    """Yield a dictionnary of tensor"""
    num_of_sequences = model_inputs["input_ids"].shape[0]
    for i in range(0, num_of_sequences, batch_size):
        batch_sequence = {key: value[i : (i + batch_size)] for key, value in model_inputs.items()}
        yield batch_sequence


def load_fasta(path_fasta: Union[str, Path]) -> List[str]:
    """Read and parse records from a fasta file

    Args:
        path_fasta: path of the fasta file

    Returns:
        List: List of sequences
    """
    if not isinstance(path_fasta, Path):
        path_fasta = Path(path_fasta).resolve()
    return [str(record.seq) for record in SeqIO.parse(str(path_fasta), format="fasta")]


def get_logs_version(path_logs):
    """Get last version of logs folder to save model inside

    Args:
        path_logs (str): path of the logs/experiments folder
    """
    version_num = None
    try:
        # Folder version organize like version_x >> catch the 'x' integer
        version = str(max([int(fold.split("_")[1]) for fold in os.listdir(path_logs)]))
        version_num = "version_" + version
    except Exception as e:
        log.debug("Found exception %s" % e)
        version_num = None
    return version_num


def format_backend(backend_list: List[str]) -> List[str]:
    """format of list to display"""
    return ["  *" + " " * 3 + model for model in backend_list]


def list_backend() -> None:
    """Get all possible backend for the model"""
    print(
        "Use backend in this list :\n\n",
        "\n".join(format_backend(BACKEND_LIST)),
        sep="",
    )
