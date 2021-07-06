"""Test module for testing loglikelihoods function"""
import numpy as np
import pytest

test_sequences = ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]
test_fasta = "data/fasta/example_fasta.fasta"
length_fasta = [538, 508, 393, 328, 354, 223]

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), "forward"),
    (2, list("ACDEFGHIKLMNPQRSTVWY") + ["MASK"], "masked"),
    (10, list("ACDEFGHIKLMNPQRSTVWY") + ["MASK"], "forward"),
]

test_params_fasta = [(1, list("ACDEFGHIKLMNPQRSTVWY"), "forward")]


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params)
def test_loglikelihoods_type_shape_and_range(init_model, batch_size, tokens_list, pass_mode):
    test_trans = init_model
    loglikelihoods = test_trans.compute_loglikelihood(
        test_sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    assert len(loglikelihoods) == len(test_sequences)
    for loglikelihood in loglikelihoods:
        assert loglikelihood <= 0 or np.isnan(loglikelihood)


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params)
def test_loglikelihoods_type_shape_and_range_fasta(init_model, batch_size, tokens_list, pass_mode):
    test_trans = init_model
    loglikelihoods = test_trans.compute_loglikelihood(
        test_fasta,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    assert len(loglikelihoods) == len(length_fasta)
    for loglikelihood in loglikelihoods:
        assert loglikelihood <= 0 or np.isnan(loglikelihood)
