"""Test module for testing accuracy function"""
import pytest
from .constants import lengths_sequence_fasta, test_fasta, test_sequences

test_params = [
    (1, "forward"),
    (2, "masked"),
    (10, "forward"),
]

test_params_fasta = [(2, "forward")]


@pytest.mark.parametrize("batch_size, pass_mode", test_params)
def test_accuracy_type_and_range(init_model, batch_size, pass_mode):
    test_trans = init_model
    accuracy = test_trans.compute_accuracy(
        test_sequences,
        batch_size=batch_size,
        pass_mode=pass_mode,
    )

    assert isinstance(accuracy, float)
    assert (accuracy >= 0.0) and (accuracy <= 1.0)


@pytest.mark.parametrize("batch_size, pass_mode", test_params_fasta)
def test_accuracy_type_and_range_fasta(init_model, batch_size, pass_mode):
    test_trans = init_model

    accuracy_fasta = test_trans.compute_accuracy(
        test_fasta,
        batch_size=batch_size,
        pass_mode=pass_mode,
    )

    assert isinstance(accuracy_fasta, float)
    assert (accuracy_fasta >= 0.0) and (accuracy_fasta <= 1.0)
