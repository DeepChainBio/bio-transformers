"""Test module for testing embeddings function"""
import torch
from biotransformers import BioTransformers

test_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]


def test_protbert_loglikelihood():
    test_trans = BioTransformers("protbert")
    loglikelihood = test_trans.compute_loglikelihoods(test_sequences)
    assert isinstance(loglikelihood, torch.Tensor)


def test_protbert_embeddings():
    test_trans = BioTransformers("protbert")
    embedding = test_trans.compute_embeddings(test_sequences)
    assert isinstance(embedding, dict)
