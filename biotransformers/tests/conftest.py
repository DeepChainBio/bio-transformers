import pytest
from biotransformers import BioTransformers

test_models = [
    # "esm1_t34_670M_UR100",
    "esm1_t6_43M_UR50S",
    # "esm1b_t33_650M_UR50S",
    "protbert",
    # "protbert_bfd",
]


@pytest.fixture(scope="session", params=test_models)
def init_model(request):
    print(request.param)
    # thanks to fixture class instance (which takes time) will be reused over tests.
    return BioTransformers(request.param)
