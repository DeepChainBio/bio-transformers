"""Test module for testing logits function"""
import pytest

test_sequences = ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]

test_params = [
    (1, "forward"),
    (2, "masked"),
    (10, "forward"),
]


@pytest.mark.parametrize("batch_size, pass_mode", test_params)
def test_logits_type(init_model, batch_size, pass_mode):
    test_trans = init_model
    logits = test_trans.compute_logits(
        test_sequences,
        batch_size=batch_size,
        pass_mode=pass_mode,
    )
    assert len(logits) == len(test_sequences)
    for logit, sequence in zip(logits, test_sequences):
        assert logit.shape[0] == len(sequence)
