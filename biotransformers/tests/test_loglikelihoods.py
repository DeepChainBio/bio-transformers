"""Test module for testing loglikelihoods function"""
import numpy as np
import pytest

test_sequences = ["AAAA", "AKKF", "AHHFK", "KKKKKKKLLL"]

test_params = [
    (1, list("ACDEFGHIKLMNPQRSTVWY"), "forward"),
    (2, ["A", "F", "K"], "masked"),
    (10, ["A"], "forward"),
]

expected = {
    "esm1_t6_43M_UR50S": [
        [-4.3960423, -10.491179, -14.885796, -9.673026],
        [-0.34790292, -6.075577, -4.432027, -0.10648791],
        [0.0, 0.0, 0.0, np.nan],
    ],
    "esm1b_t33_650M_UR50S": [
        [-0.8544625, -3.2860425, -3.916751, -9.909236],
        [-1.903336, -5.403609, -4.2642317, -1.3674309],
        [0.0, 0.0, 0.0, np.nan],
    ],
    "protbert": [
        [-5.140623, -6.5139484, -1.9810636, -56.23104],
        [-3.6754684, -5.9086, -6.6085267, -8.540686],
        [0.0, 0.0, 0.0, np.nan],
    ],
}


# expected = {
#     "esm1_t6_43M_UR50S": [-118.64253, -141.12074],
#     "esm1b_t33_650M_UR50S": [-13.180167, -15.423159],
#     "protbert": [-12.557711, -21.82305],
# }


@pytest.mark.parametrize("batch_size, tokens_list, pass_mode", test_params)
def test_loglikelihoods_type_shape_and_range(
    init_model, batch_size, tokens_list, pass_mode
):
    test_trans = init_model
    loglikelihoods = test_trans.compute_loglikelihood(
        test_sequences,
        batch_size=batch_size,
        tokens_list=tokens_list,
        pass_mode=pass_mode,
    )
    print(
        "batch_size:", batch_size, "tokens_list:", tokens_list, "pass_mode:", pass_mode
    )
    print(loglikelihoods)
    assert isinstance(loglikelihoods, np.ndarray)
    assert loglikelihoods.shape == (len(test_sequences),)
    for loglikelihood in loglikelihoods:
        assert loglikelihood <= 0 or np.isnan(loglikelihood)
