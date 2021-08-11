"""Test module for testing loglikelihoods function"""
import pytest
from numpy.testing import assert_allclose

test_params = [
    ([["A1Q"], ["A1K", "K2A"], ["A1H"], ["K3W", "K2D", "L9H"]], "params1"),
]


@pytest.mark.parametrize("mutations, params", test_params)
def test_mutation_score_type_shape_and_range(
    init_model, sequences, mutations_score_results, mutations, params
):
    test_trans = init_model
    mutations_scores = test_trans.compute_mutation_score(sequences, mutations)
    assert len(mutations_scores) == len(sequences)
    if test_trans._model_dir in mutations_score_results.keys():
        results = mutations_score_results[test_trans._model_dir][params]
        assert_allclose(mutations_scores, results, rtol=0.01)
