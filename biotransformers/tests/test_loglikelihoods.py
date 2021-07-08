"""Test module for testing loglikelihoods function"""
import numpy as np
import pytest

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), "forward"),
    (2, list("ACDEFGHIKLMNPQRSTVWY") + ["MASK"], "masked"),
    (10, list("ACDEFGHIKLMNPQRSTVWY") + ["MASK"], "forward"),
]

test_params_fasta = [(1, list("DASFIHEMKLXQPWYRNTGVC"), "forward")]


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params)
def test_loglikelihoods_type_shape_and_range(
    init_model, sequences, batch_size, tokens_list, pass_mode
):
    test_trans = init_model
    loglikelihoods = test_trans.compute_loglikelihood(
        sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    assert len(loglikelihoods) == len(sequences)
    for loglikelihood in loglikelihoods:
        assert loglikelihood <= 0 or np.isnan(loglikelihood)


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params_fasta)
def test_loglikelihoods_type_shape_and_range_fasta(
    init_model, fasta_path, lengths_sequence_fasta, batch_size, tokens_list, pass_mode
):
    test_trans = init_model
    loglikelihoods = test_trans.compute_loglikelihood(
        fasta_path,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    assert len(loglikelihoods) == len(lengths_sequence_fasta)
    for loglikelihood in loglikelihoods:
        assert loglikelihood <= 0 or np.isnan(loglikelihood)
