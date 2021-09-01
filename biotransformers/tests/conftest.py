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
    lengths = [476, 201, 60, 35, 284]
    return lengths


@pytest.fixture(scope="session")
def loglikelihoods_results():
    results = {
        "esm1_t6_43M_UR50S": {
            "params1": [
                -1.0990107895886871,
                -2.622793987825229,
                -2.9771586269887687,
                -0.9673027336521154,
            ],
            "params2": [
                -1.025939130590268,
                -3.251339913625416,
                -3.471589807380847,
                -0.7787956539618508,
            ],
            "params3": [
                -1.099011026298114,
                -2.6227955882668903,
                -2.9771586069197626,
                -0.9673018411528049,
            ],
        },
        "Rostlab/prot_bert": {
            "params1": [
                -1.2851557324014358,
                -1.6284870947647039,
                -0.3962127573832269,
                -5.623104014419566,
            ],
            "params2": [
                -3.314856902784298,
                -3.7017992105942645,
                -3.5632398628301134,
                -4.201843982611495,
            ],
            "params3": [
                -1.2851545893520187,
                -1.6284828574671406,
                -0.3962134847024349,
                -5.623108921843686,
            ],
        },
    }
    return results


@pytest.fixture(scope="session")
def loglikelihoods_fasta_results():
    results = {
        "esm1_t6_43M_UR50S": [
            -3.088591489364207,
            -2.258382942624286,
            -1.8204519701969795,
            -1.712802577224046,
            -1.0209009845359696,
        ],
        "Rostlab/prot_bert": [
            -0.13746537830140257,
            -0.2248445733096602,
            -0.3628546008213477,
            -0.2788796518420453,
            -0.32384316791422224,
        ],
    }
    return results


@pytest.fixture(scope="session")
def mutations_score_results():
    results = {
        "esm1_t6_43M_UR50S": {
            "params1": [
                -2.522218942642212,
                0.9405336380004883,
                0.2436962127685547,
                -17.050978302955627,
            ],
        },
        "Rostlab/prot_bert": {
            "params1": [
                -0.496506929397583,
                1.3073345646262169,
                0.24045005440711975,
                -2.4403315782547,
            ]
        },
    }
    return results
