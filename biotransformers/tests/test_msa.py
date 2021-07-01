from glob import glob

import numpy as np
from biotransformers import BioTransformers

path_msa = "data/msa"
model = BioTransformers("esm_msa1_t12_100M_UR50S")


def test_msa_embeddings_type_and_shape():
    n_seqs_msa = 6
    embeddings = model.compute_embeddings(path_msa, n_seqs_msa=n_seqs_msa)
    assert isinstance(embeddings, dict)
    assert isinstance(embeddings["cls"], np.ndarray)
    assert embeddings["cls"].shape[0] == len(glob(path_msa + "/*.a3m"))
    assert embeddings["cls"].shape[1] == n_seqs_msa


def test_msa_logits_type():
    n_seqs_msa = 6
    logits = model.compute_logits(path_msa, n_seqs_msa=n_seqs_msa)

    assert len(logits) == len(glob(path_msa + "/*.a3m"))
    for logit in logits:
        assert logit.shape[0] == n_seqs_msa
