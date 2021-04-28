"""Test module for testing loglikelihood function"""
from biotransformers import BioTransformers

test_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]


def test_esm_loglikelihood():
    test_trans = BioTransformers("esm1_t34_670M_UR100")
    loglikelihood = test_trans.compute_loglikelihood(test_sequences)
    assert isinstance(loglikelihood, list)


def test_protbert_loglikelihood():
    test_trans = BioTransformers("protbert")
    loglikelihood = test_trans.compute_loglikelihood(test_sequences)
    assert isinstance(loglikelihood, list)
