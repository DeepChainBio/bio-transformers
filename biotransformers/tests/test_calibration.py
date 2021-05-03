"""Test module for testing calibration function"""
import pytest

test_sequences = ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), "forward"),
    (2, ["A", "F", "K"], "masked"),
    (10, ["A"], "forward"),
]


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params)
def test_calibration_type_and_range(init_model, batch_size, tokens_list, pass_mode):
    test_trans = init_model
    results = test_trans.compute_calibration(
        test_sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    accuracy = results["accuracy"]
    assert isinstance(accuracy, float)
    assert (accuracy >= 0.0) and (accuracy <= 1.0)
    ece = results["ece"]
    assert isinstance(ece, float)
    assert (ece >= 0.0) and (ece <= 1.0)
    reliability_diagram = results["reliability_diagram"]
    assert isinstance(reliability_diagram, list)
    for value in reliability_diagram:
        assert isinstance(value, float)
        assert (value >= 0.0) and (value <= 1.0)
