import re
from typing import Dict, List, Tuple

import numpy as np
from biotransformers.utils.constant import NATURAL_AAS_LIST

ProbTuple = Tuple[float, float]
TokenProbsDict = Dict[int, Dict[str, float]]
SequenceProbsList = List[TokenProbsDict]


class InvalidPositionStringError(Exception):
    """Raised when a position string is passed with incorrect format"""


def validate_position_str(position_str: str):
    """Checks positions str format"""
    regex = re.compile(r"[A-Z]{1}[0-9]{1,4}[A-Z]{1}", re.I)
    if not regex.match(position_str):
        raise InvalidPositionStringError(
            f"'{position_str}' is not a valid position string"
        )


class Mutation:
    """register a mutation from a string

    Args:
        mutation (str): string mutation format "A8U" -> "NativeIdMutant"
    """

    def __init__(self, mutation_str: str) -> None:
        validate_position_str(mutation_str)
        self.mutation_str = mutation_str
        self.mutation = mutation_str[-1]
        self.native = mutation_str[0]
        self.position = int(mutation_str[1:-1])

    def __repr__(self) -> str:
        return f"Mutation >> Native: {self.native} New: {self.mutation} at position {self.position}"

    def is_valid_mutation(self, sequence: str):
        """Check if mutation is valid for the sequence of AA
        Args:
            sequence (str): protein sequence string
        """
        if len(sequence) < self.position:
            raise ValueError(
                f"Sequence smaller than position {self.position} for mutation {self.mutation_str}"
            )
        if self.native != sequence[self.position - 1]:
            raise InvalidPositionStringError(
                f"'{self.native}' is not a valid native position for mutation {self.mutation_str}"
            )
        if self.mutation not in NATURAL_AAS_LIST:
            raise ValueError(
                f"New amino acid {self.mutation} is not a valid mutation for {self.mutation_str}"
            )
        return True

def get_list_probs(
    mutation_list: List[Tuple[Mutation]],
    mutate_probs: SequenceProbsList,
    length_mutations: List[int],
) -> Tuple[List[List[float]], List[List[float]]]:
    """This function build a list of mutate and native probabilities to compute
    the mutate_score. For each position in the mutate list, we catch the native probability
    and the mutate probability of this position. We do this for each sequence and return two
    lists : native_probs and mutate probs.

    Args:
        mutation_list (List[Mutation]): list with integer which are mutations
        mutate_probs (List[Dict[Any]]): probabilities for mutate sequence
        length_mutations (List[int]):  length of indivual mutation for each sequence
    """
    flat_mutation = [mut for tup in mutation_list for mut in tup]
    native_probs_list, mutate_probs_list = [], []
    for prob, mut in zip(mutate_probs, flat_mutation):
        native_probs_list.append(prob[mut.position - 1][mut.native])
        mutate_probs_list.append(prob[mut.position - 1][mut.mutation])
    return split_list(native_probs_list, length_mutations), split_list(
        mutate_probs_list, length_mutations
    )


def mutation_score(native_probs: List[float], mutate_probs: List[float]) -> float:
    """
    Compute mutate score based on Masked marginal probability
    Sum(log(p(xi=xi_mutate|x-M))-log(p(xi=xi_native|x-M))) over M (M s a mutation set)

    Args:
        native_probs (List[ProbTuple]): [description]
        mutate_probs (List[ProbTuple]): [description]

    Returns:
        List[float]: [description]
    """
    return np.sum(
        [np.log(m_p) - np.log(n_p) for m_p, n_p in zip(mutate_probs, native_probs)]
    )


def split_list(list_to_split: List, lengths_list: List) -> List[List]:  # type: ignore
    """split a list in sublist

    Args:
        list_to_split (List): native list
        lengths_list (List): length of each sublist

    Returns:
        [type]: List of sublist
    """
    assert len(list_to_split) == sum(
        lengths_list
    ), "Sum of sublist length is not valid."
    splitted_list = []
    count = 0
    for length in lengths_list:
        splitted_list.append(list_to_split[count : (count + length)])
        count += length
    return splitted_list
