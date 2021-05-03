"""Test module for testing logits function"""
import numpy as np
import pytest

test_sequences = ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), "forward"),
    (2, ["A", "F", "K"], "masked"),
    (10, ["A"], "forward"),
]


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params)
def test_logits_type(init_model, batch_size, tokens_list, pass_mode):
    test_trans = init_model
    logits, labels = test_trans.compute_logits(
        test_sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    assert isinstance(logits, np.ndarray)
    assert isinstance(labels, np.ndarray)
