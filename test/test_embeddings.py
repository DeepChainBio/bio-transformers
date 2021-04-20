"""Test module for testing embeddings function"""
from biotransformers import BioTransformers

test_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]


def test_esm_embeddings():
    test_trans = BioTransformers("esm1_t34_670M_UR100")
    embedding = test_trans.compute_embeddings(test_sequences)
    assert isinstance(embedding, dict)


def test_protbert_embeddings():
    test_trans = BioTransformers("protbert")
    embedding = test_trans.compute_embeddings(test_sequences)
    assert isinstance(embedding, dict)
