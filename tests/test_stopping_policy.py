from __future__ import annotations
import numpy as np
import pytest
from src.sigstop.stopping.policy import (
    LinearStoppingPolicy,
    initialize_linear_policy,
    initialize_policy_from_feature_tensor,
    score_signature_features,
)


def test_policy_scores_single_path_features_without_bias_by_default() -> None:
    features = np.array(
        [
            [1.0, 2.0, -1.0],
            [0.5, 0.0, 2.0],
        ],
        dtype = np.float64,
    )
    policy = LinearStoppingPolicy(
        weights = np.array([0.2, -0.5, 1.0], dtype = np.float64),
    )

    scores = score_signature_features(policy, features)

    expected = features @ policy.weights
    np.testing.assert_allclose(scores, expected)
    assert policy.use_bias is False


def test_policy_scores_batched_features_with_bias_when_enabled() -> None:
    features = np.array(
        [
            [[1.0, 0.0], [0.5, 1.0]],
            [[0.0, 1.0], [2.0, -1.0]],
        ],
        dtype = np.float64,
    )
    policy = LinearStoppingPolicy(
        weights = np.array([2.0, -1.0], dtype = np.float64),
        bias = 0.25,
        use_bias = True,
    )

    scores = score_signature_features(policy, features)

    expected = np.tensordot(features, policy.weights, axes = ([-1], [0])) + 0.25
    np.testing.assert_allclose(scores, expected)


def test_policy_dimension_mismatch_raises_clear_error() -> None:
    features = np.ones((3, 4), dtype = np.float64)
    policy = LinearStoppingPolicy(weights = np.ones(5, dtype = np.float64))

    with pytest.raises(ValueError, match = "input dimension"):
        score_signature_features(policy, features)


def test_small_normal_init_is_seeded_and_nontrivial() -> None:
    policy_a = initialize_linear_policy(
        4,
        include_bias = False,
        init = "small_normal",
        weight_scale = 1.0e-2,
        seed = 123,
        dtype = np.float32,
    )
    policy_b = initialize_linear_policy(
        4,
        include_bias = False,
        init = "small_normal",
        weight_scale = 1.0e-2,
        seed = 123,
        dtype = np.float32,
    )

    np.testing.assert_allclose(policy_a.weights, policy_b.weights)
    assert np.linalg.norm(policy_a.weights) > 0.0
    assert policy_a.use_bias is False
    assert policy_a.scalar_term_in_signatures is True


def test_initialize_policy_from_feature_tensor_infers_last_dimension() -> None:
    features = np.zeros((5, 7, 31), dtype = np.float32)

    policy = initialize_policy_from_feature_tensor(
        features,
        include_bias = False,
        init = "zeros",
        dtype = np.float32,
    )

    assert policy.input_dim == 31
    assert policy.weights.shape == (31,)
    np.testing.assert_allclose(policy.weights, np.zeros(31, dtype = np.float32))


def test_policy_serialization_metadata_is_consistent() -> None:
    policy = LinearStoppingPolicy(
        weights = np.array([1.0, -2.0, 3.5], dtype = np.float32),
        bias = 0.0,
        use_bias = False,
        scalar_term_in_signatures = True,
        init = "small_normal",
    )

    payload = policy.to_dict()

    assert payload["input_dim"] == 3
    assert payload["use_bias"] is False
    assert payload["scalar_term_in_signatures"] is True
    assert payload["dtype"] == "float32"
