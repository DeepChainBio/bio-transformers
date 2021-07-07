import pytest
from biotransformers import BioTransformers

test_models = [
    # "esm1_t34_670M_UR100",
    "esm1_t6_43M_UR50S",
    # "esm1b_t33_650M_UR50S",
    # "protbert",
    # "protbert_bfd",
]


@pytest.fixture(scope="session", params=test_models)
def init_model(request):
    print(request.param)
    # thanks to fixture class instance (which takes time) will be reused over tests.
    return BioTransformers(request.param)


@pytest.fixture(scope="session")
def fasta_path():
    # thanks to fixture class instance (which takes time) will be reused over tests.
    return "data/fasta/example_fasta.fasta"


@pytest.fixture(scope="session")
def sequences():
    # thanks to fixture class instance (which takes time) will be reused over tests.
    return ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]


@pytest.fixture(scope="session")
def lengths_sequence_fasta():
    # thanks to fixture class instance (which takes time) will be reused over tests.
    lengths_sequence_fasta = [
        476,
        201,
        60,
        35,
        500,
        284,
        316,
        35,
        36,
        502,
        158,
        50,
        670,
        300,
        325,
        258,
        48,
        35,
        80,
        31,
        31,
        89,
        90,
        418,
        149,
        45,
        103,
        103,
        103,
        103,
        105,
        105,
        231,
        384,
        45,
        222,
        310,
        87,
        45,
        258,
        340,
        249,
        51,
        215,
        98,
        40,
        56,
        96,
        280,
        22,
        57,
        139,
        93,
        80,
        146,
        50,
        52,
        52,
        51,
        51,
        81,
        52,
        51,
        52,
        83,
        43,
        43,
        43,
        445,
        23,
        75,
        51,
        50,
        113,
        198,
        97,
        72,
        289,
        169,
    ]
    return lengths_sequence_fasta
