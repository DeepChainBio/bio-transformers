"""Test module for testing accuracy function"""
import pytest

test_sequences = ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), "forward"),
    (2, ["A", "F", "K"], "masked"),
    (10, ["A"], "forward"),
]


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params)
def test_accuracy_type_and_range(init_model, batch_size, tokens_list, pass_mode):
    test_trans = init_model
    accuracy = test_trans.compute_accuracy(
        test_sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    assert isinstance(accuracy, float)
    assert (accuracy >= 0.0) and (accuracy <= 1.0)
