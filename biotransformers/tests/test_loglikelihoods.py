"""Test module for testing loglikelihoods function"""
import pytest
from numpy.testing import assert_allclose

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), "forward", "params1"),
    (2, list("ACDEFGHIKLMNPQRSTVWY") + ["MASK"], "masked", "params2"),
    (10, list("ACDEFGHIKLMNPQRSTVWY") + ["MASK"], "forward", "params3"),
]

test_params_fasta = [(1, list("KFQRVACEXWIHYPNGSMTDL"), "forward")]


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode, params", test_params)
def test_loglikelihoods_type_shape_and_range(
    init_model,
    sequences,
    loglikelihoods_results,
    batch_size,
    tokens_list,
    pass_mode,
    params,
):
    test_trans = init_model
    loglikelihoods = test_trans.compute_loglikelihood(
        sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
        normalize=True,
    )
    assert len(loglikelihoods) == len(sequences)
    if test_trans._model_dir in loglikelihoods_results.keys():
        results = loglikelihoods_results[test_trans._model_dir][params]
        assert_allclose(loglikelihoods, results, rtol=0.01)


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params_fasta)
def test_loglikelihoods_type_shape_and_range_fasta(
    init_model,
    fasta_path,
    lengths_sequence_fasta,
    loglikelihoods_fasta_results,
    batch_size,
    tokens_list,
    pass_mode,
):
    test_trans = init_model
    loglikelihoods = test_trans.compute_loglikelihood(
        fasta_path,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
        normalize=True,
    )
    assert len(loglikelihoods) == len(lengths_sequence_fasta)
    if test_trans._model_dir in loglikelihoods_fasta_results.keys():
        results = loglikelihoods_fasta_results[test_trans._model_dir]
        assert_allclose(loglikelihoods, results, rtol=0.01)
