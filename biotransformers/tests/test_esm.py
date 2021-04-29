"""Test module for testing loglikelihood function"""
import numpy as np
from biotransformers import BioTransformers

test_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]


def test_esm_loglikelihood():
    test_trans = BioTransformers("esm1_t34_670M_UR100")
    loglikelihood = test_trans.compute_loglikelihoods(test_sequences)
    assert isinstance(loglikelihood, np.ndarray)


def test_esm_embeddings():
    test_trans = BioTransformers("esm1_t34_670M_UR100")
    embedding = test_trans.compute_embeddings(test_sequences)
    assert isinstance(embedding, dict)
