"""Test module for testing embeddings function"""
from biotransformers import BioTransformers

test_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]


def test_esm_embeddings():
    test_trans = BioTransformers()
    embedding = test_trans.compute_embeddings(test_sequences)
    assert isinstance(embedding, dict)
