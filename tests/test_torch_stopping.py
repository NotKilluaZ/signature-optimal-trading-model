from __future__ import annotations
import numpy as np
import torch
from src.sigstop.stopping.loss import smoothed_stopping_loss
from src.sigstop.stopping.policy import LinearStoppingPolicy
from src.sigstop.stopping.torch_loss import build_torch_smoothed_loss_result
from src.sigstop.stopping.torch_policy import build_torch_policy_from_linear_policy


def test_torch_policy_scores_match_numpy_policy_scores() -> None:
    features = np.array(
        [
            [[1.0, 2.0], [0.5, -1.0]],
            [[0.0, 3.0], [2.0, 1.5]],
        ],
        dtype = np.float32,
    )
    policy = LinearStoppingPolicy(
        weights = np.array([0.25, -0.5], dtype = np.float32),
        bias = 0.1,
        use_bias = True,
    )

    torch_policy = build_torch_policy_from_linear_policy(policy, device = "cpu")
    torch_scores = torch_policy(torch.as_tensor(features)).detach().cpu().numpy()
    numpy_scores = np.tensordot(features, policy.weights, axes = ([-1], [0])) + policy.bias

    np.testing.assert_allclose(torch_scores, numpy_scores)


def test_torch_smoothed_loss_matches_numpy_smoothed_loss() -> None:
    scores = np.array(
        [
            [0.1, -0.05, 0.08],
            [0.02, 0.04, -0.03],
        ],
        dtype = np.float32,
    )
    payoffs = np.array(
        [
            [0.0, -0.2, -0.5, -0.7],
            [0.0, 0.1, 0.05, -0.1],
        ],
        dtype = np.float32,
    )

    torch_result = build_torch_smoothed_loss_result(
        torch.as_tensor(scores),
        torch.as_tensor(payoffs),
        threshold = 0.05,
        mu = 20.0,
    )
    numpy_loss = smoothed_stopping_loss(scores, payoffs, threshold = 0.05, mu = 20.0)

    np.testing.assert_allclose(torch_result.loss.detach().cpu().item(), numpy_loss, rtol = 1.0e-6, atol = 1.0e-6)
