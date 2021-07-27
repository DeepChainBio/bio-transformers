from typing import Dict, List, Tuple

import numpy as np

ProbTuple = Tuple[float, float]
TokenProbsDict = Dict[int, Dict[str, float]]
SequenceProbsList = List[TokenProbsDict]


def get_list_probs(
    sequences: List[str], mutation_list: List[int], mutate_probs: SequenceProbsList
) -> Tuple[List[ProbTuple], List[ProbTuple]]:
    """This function build a list of mutate and native probabilities to compute
    the mutate_score. For each position in the mutate list, we catch the native probability
    and the mutate probability of this position. We do this for each sequence and return two
    lists : native_probs and mutate probs.

    Args:
        sequences (List[str]): list of protein sequence
        mutation_list (List[int]): list with integer which are mutations
        mutate_probs (List[Dict[Any]]): [description]

    """
    index_1, index_2 = mutation_list
    # change to python 0 indexing
    index_1 -= 1
    index_2 -= 1
    native_probs_list, mutate_probs_list = [], []
    for prob, seq in zip(mutate_probs, sequences):
        native_probs_list.append((prob[index_1][seq[index_1]], prob[index_2][seq[index_2]]))
        mutate_probs_list.append((prob[index_1][seq[index_2]], prob[index_2][seq[index_1]]))
    return native_probs_list, mutate_probs_list


def mutation_score(native_probs: ProbTuple, mutate_probs: ProbTuple) -> List[float]:
    """
    Compute mutate score based on Masked marginal probability
    Sum(log(p(xi=xi_mutate|x-M))-log(p(xi=xi_native|x-M))) over M (M s a mutation set)

    Args:
        native_probs (List[ProbTuple]): [description]
        mutate_probs (List[ProbTuple]): [description]

    Returns:
        List[float]: [description]
    """
    return np.sum([np.log(m_p) - np.log(n_p) for m_p, n_p in zip(mutate_probs, native_probs)])
