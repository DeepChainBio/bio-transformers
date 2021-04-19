"""Test module for testing loglikelihood function"""
from biotransformers import BioTransformers

test_sequences = [
    "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
    "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE",
]


def test_esm_loglikelihood():
    test_trans = BioTransformers()
    loglikelihood = test_trans.compute_loglikelihood(test_sequences)
    assert isinstance(loglikelihood, list)
