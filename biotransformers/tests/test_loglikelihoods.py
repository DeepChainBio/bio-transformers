"""Test module for testing loglikelihoods function"""
import numpy as np
import pytest

test_sequences = ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), "forward"),
    (2, ["A", "F", "K"], "masked"),
    (10, ["A"], "forward"),
]


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params)
def test_loglikelihoods_type_shape_and_range(
    init_model, batch_size, tokens_list, pass_mode
):
    test_trans = init_model
    loglikelihoods = test_trans.compute_loglikelihood(
        test_sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    assert isinstance(loglikelihoods, np.ndarray)
    assert loglikelihoods.shape == (len(test_sequences),)
    for loglikelihood in loglikelihoods:
        assert loglikelihood <= 0 or np.isnan(loglikelihood)
