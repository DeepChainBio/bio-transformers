"""Test module for testing embeddings function"""
import numpy as np
import pytest

test_sequences = ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), ["cls", "mean"]),
    (2, ["A", "F", "K"], ["min", "max", "cls"]),
    (10, ["A"], ["cls", "mean"]),
]


@pytest.mark.parametrize("batch_size, tokens_list, pool_mode", test_params)
def test_embeddings_type_and_shape(init_model, batch_size, tokens_list, pool_mode):
    test_trans = init_model
    embeddings = test_trans.compute_embeddings(
        test_sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pool_mode=pool_mode,
    )
    assert isinstance(embeddings, dict)
    for emb in embeddings.values():
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (len(test_sequences), test_trans.embeddings_size)
