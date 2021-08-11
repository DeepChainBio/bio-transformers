"""Test module for testing embeddings function"""
import numpy as np
import pytest

test_params = [
    (1, ["cls", "mean"]),
    (2, ["full", "mean", "cls"]),
    (10, ["cls", "full"]),
]


@pytest.mark.parametrize("batch_size, pool_mode", test_params)
def test_embeddings_type_and_shape(init_model, sequences, batch_size, pool_mode):
    test_trans = init_model
    embeddings = test_trans.compute_embeddings(
        sequences,
        batch_size=batch_size,
        pool_mode=pool_mode,
    )

    assert isinstance(embeddings, dict)
    if "full" in pool_mode:
        for emb, sequence in zip(embeddings["full"], sequences):
            assert emb.shape[0] == len(sequence)
    if "cls" in pool_mode:
        assert isinstance(embeddings["cls"], np.ndarray)

    if "mean" in pool_mode:
        assert isinstance(embeddings["mean"], np.ndarray)


@pytest.mark.parametrize("batch_size, pool_mode", test_params)
def test_embeddings_type_and_shape_fasta(
    init_model, fasta_path, lengths_sequence_fasta, batch_size, pool_mode
):
    test_trans = init_model
    embeddings = test_trans.compute_embeddings(
        fasta_path,
        batch_size=batch_size,
        pool_mode=pool_mode,
    )
    if "full" in pool_mode:
        for emb, length in zip(embeddings["full"], lengths_sequence_fasta):
            assert emb.shape[0] == length

    assert isinstance(embeddings, dict)
    if "cls" in pool_mode:
        assert isinstance(embeddings["cls"], np.ndarray)

    if "mean" in pool_mode:
        assert isinstance(embeddings["mean"], np.ndarray)
