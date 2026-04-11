from __future__ import annotations
import numpy as np
from src.sigstop.stopping.loss import build_smoothed_loss_result, smoothed_stopping_loss
from src.sigstop.stopping.payoff import (
    build_entry_payoff_process,
    build_exit_payoff_process,
    compute_payoff_increments,
)
from src.sigstop.stopping.stop_rule import (
    apply_deterministic_stop_rule,
    compute_cumulative_squared_scores,
    find_batched_first_threshold_crossings,
    find_first_threshold_crossing,
)


def _finite_difference_gradient(
    func,
    x: np.ndarray,
    *,
    eps: float = 1e-6,
) -> np.ndarray:
    grad = np.empty_like(x, dtype = np.float64)

    for index in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[index] += eps
        x_minus[index] -= eps
        grad[index] = (func(x_plus) - func(x_minus)) / (2.0 * eps)

    return grad


def test_cumulative_squared_scores_are_monotone() -> None:
    scores = np.array([0.2, -0.5, 0.1], dtype = np.float64)

    cumulative = compute_cumulative_squared_scores(scores)

    np.testing.assert_allclose(cumulative, np.array([0.04, 0.29, 0.30]))
    assert np.all(np.diff(cumulative) >= 0.0)


def test_stop_rule_uses_first_greater_equal_crossing() -> None:
    scores = np.array([0.1, 0.2, 0.1], dtype = np.float64)

    result = apply_deterministic_stop_rule(scores, threshold = 0.05)

    np.testing.assert_allclose(result.cumulative_squared_scores, np.array([0.01, 0.05, 0.06]))
    assert result.stop_index == 1
    assert result.crossed is True
    assert find_first_threshold_crossing(result.cumulative_squared_scores, 0.05) == 1


def test_stop_rule_returns_none_when_threshold_is_never_crossed() -> None:
    scores = np.array([0.05, -0.02, 0.03], dtype = np.float64)

    result = apply_deterministic_stop_rule(scores, threshold = 0.5)

    assert result.stop_index is None
    assert result.crossed is False


def test_batched_crossings_mark_missing_paths() -> None:
    cumulative = np.array(
        [
            [0.01, 0.04, 0.09],
            [0.002, 0.003, 0.004],
        ],
        dtype = np.float64,
    )

    crossings = find_batched_first_threshold_crossings(cumulative, threshold = 0.05)

    np.testing.assert_array_equal(crossings, np.array([2, -1], dtype = np.int32))


def test_entry_and_exit_payoff_processes_match_paper_formulas() -> None:
    spread = np.array([2.0, 1.0, 3.0], dtype = np.float64)
    times = np.array([0.0, 1.0, 2.0], dtype = np.float64)

    entry = build_entry_payoff_process(spread, r = 0.1, c = 0.5, times = times)
    exit_ = build_exit_payoff_process(spread, r_hat = 0.1, c_hat = 0.25, times = times)

    expected_discount = np.exp(-0.1 * times)
    np.testing.assert_allclose(entry, expected_discount * (-spread - 0.5))
    np.testing.assert_allclose(exit_, expected_discount * (spread - 0.25))
    np.testing.assert_allclose(
        compute_payoff_increments(exit_),
        np.diff(expected_discount * (spread - 0.25)),
    )


def test_smoothed_loss_matches_manual_eq19_style_computation() -> None:
    scores = np.array([0.1, 0.2], dtype = np.float64)
    payoffs = np.array([1.0, 1.5, 1.2], dtype = np.float64)

    result = build_smoothed_loss_result(
        scores,
        payoffs,
        threshold = 0.05,
        mu = 20.0,
    )

    cumulative = np.array([0.01, 0.05], dtype = np.float64)
    stop_gate = 1.0 / (1.0 + np.exp(-20.0 * (cumulative - 0.05)))
    survival = 1.0 - stop_gate
    manual_value = payoffs[0] + np.sum(survival * np.diff(payoffs))
    manual_loss = -manual_value

    np.testing.assert_allclose(result.cumulative_squared_scores, cumulative)
    np.testing.assert_allclose(result.stop_indicator, stop_gate)
    np.testing.assert_allclose(result.survival_indicator, survival)
    np.testing.assert_allclose(result.loss, manual_loss)


def test_smoothed_loss_has_finite_numerical_gradients() -> None:
    scores = np.array([0.09, -0.08, 0.06], dtype = np.float64)
    payoffs = np.array([0.0, 0.25, -0.1, 0.35], dtype = np.float64)

    def objective(local_scores: np.ndarray) -> float:
        return smoothed_stopping_loss(
            local_scores,
            payoffs,
            threshold = 0.05,
            mu = 20.0,
        )

    gradient = _finite_difference_gradient(objective, scores)

    assert np.all(np.isfinite(gradient))
    assert float(np.linalg.norm(gradient)) > 0.0


def test_smoothed_loss_supports_batched_inputs() -> None:
    scores = np.array(
        [
            [0.05, 0.10],
            [0.04, -0.03],
        ],
        dtype = np.float64,
    )
    payoffs = np.array(
        [
            [0.0, 0.1, 0.2],
            [0.0, -0.1, 0.05],
        ],
        dtype = np.float64,
    )

    result = build_smoothed_loss_result(scores, payoffs, threshold = 0.05, mu = 20.0)

    assert result.cumulative_squared_scores.shape == (2, 2)
    assert result.path_values.shape == (2,)
    assert np.isfinite(result.loss)
