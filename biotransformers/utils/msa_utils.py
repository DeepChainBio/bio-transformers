import itertools
import os
import string
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from Bio import SeqIO


def get_translation() -> Dict[int, Any]:
    """
    get translation dict to convert unused character in MSA
    """
    delete_keys = dict.fromkeys(string.ascii_lowercase)
    delete_keys["."] = None  # gap character  '-' alignement character
    delete_keys["*"] = None
    translation = str.maketrans(delete_keys)
    return translation


def read_sequence(filename: str) -> Tuple[str, str]:
    """Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)


def remove_insertions(sequence: str) -> str:
    """Removes any insertions into the sequence.
    Needed to load aligned sequences in an MSA."""
    translation = get_translation()
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """Reads the first nseq sequences from an MSA file,
    automatically removes insertions."""
    return [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]


def get_msa_list(path_msa: Optional[str]) -> List[str]:
    """Get all files of the msa folder and check file format

    Args:
        path_msa (Optional[str]): path of the folder with a3m file
    """
    if path_msa is None:
        raise ValueError("The path of the msa folder could not be None with msa model.")
    if not os.path.isdir(path_msa):
        raise FileExistsError(f"{path_msa} is not a valid directory")

    list_msa = glob(path_msa + "/*.a3m")
    all_a3m_file = all([msa.endswith("a3m") for msa in list_msa])
    if len(list_msa) == 0:
        raise FileNotFoundError(
            "Can't find any msa files with .a3m format in this folder."
        )
    if not all_a3m_file:
        raise ValueError("All files in msa folder should have a3m format.")

    return list_msa


def get_msa_lengths(list_msa: List[List[Tuple[str, str]]], nseq: int) -> List[int]:
    """Get length of an MSA list

    All MSA must have at least nseq in msa

    Args:
        list_msa (List[List[Tuple[str,str]]]): list of MSA. MSA is a list of tuple
        nseq
    Returns:
        List[int]: [description]
    """

    def _msa_length(msa: List[Tuple[str, str]]) -> List[int]:
        """get length of each sequence in msa

        Example: >> input = ['AAAB','AAAA','AAA-']
                 >> _msa_length(input)
                 >> [4,4,4]
        Raises:
            ValueError if number of seq in the MSA is less than nseq
        Args:
            msa (List[Tuple[str, str]]): List of sequence

        Returns:
            List[int]: List of length of each msa
        """
        return [len(seq[1]) for seq in msa]

    lengths = [_msa_length(msa) for msa in list_msa]
    n_different_seq = sum([len(length) != nseq for length in lengths])
    if n_different_seq > 0:
        msg = (
            f"Find {n_different_seq} files with less than {nseq} sequences in the msa. "
            f"All msa files must have at least {nseq} sequences. "
            f"Use `from biotransformers.utils.msa_utils.msa_to_remove` to get the file to remove."
        )
        raise ValueError(msg)
    unique_length = [max(length) for length in lengths]
    return unique_length


def msa_to_remove(path_msa: str, n_seq) -> List[str]:
    """Get list of msa with less than nseq sequence

    Args:
        path_msa (str): [description]

    Returns:
        List of msa filepath that don't have enough enough sequences.
    """
    path_msa = str(Path(path_msa).resolve())
    list_msa_filepath = get_msa_list(path_msa)
    list_msa = [read_msa(file, n_seq) for file in list_msa_filepath]

    def _msa_length(msa: List[Tuple[str, str]]) -> List[int]:
        return [len(seq[1]) for seq in msa]

    lengths = [_msa_length(msa) for msa in list_msa]
    msa_to_remove = []
    for i, length in enumerate(lengths):
        if len(length) != n_seq:
            msa_to_remove.append(list_msa_filepath[i])
    print(
        f"{len(msa_to_remove)}/{len(list_msa)} have insufficient number of sequences in MSA."
    )
    return msa_to_remove
