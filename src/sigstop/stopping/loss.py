from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from src.sigstop.stopping.payoff import compute_payoff_increments
from src.sigstop.stopping.stop_rule import compute_batched_cumulative_squared_scores


@dataclass(frozen = True)
class SmoothedLossResult:
    scores: np.ndarray
    cumulative_squared_scores: np.ndarray
    stop_indicator: np.ndarray
    survival_indicator: np.ndarray
    payoff_increments: np.ndarray
    path_values: np.ndarray
    loss: float

# Convert input values into a validated batched 2D array
def _coerce_batched_array(
    values: np.ndarray | list[float] | list[list[float]],
    *,
    name: str,
) -> tuple[np.ndarray, bool]:
    array = np.asarray(values, dtype = np.float64)
    squeezed = False

    if array.ndim == 1:
        array = array[None, :]
        squeezed = True
    elif array.ndim != 2:
        raise ValueError(
            f"{name} must have shape (T,) or (B, T). Got shape: {array.shape}"
        )

    if array.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite value(s).")

    return array, squeezed

# Match score and payoff batch dimensions for paired computation
def _broadcast_batch_dimension(
    scores: np.ndarray,
    payoffs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if scores.shape[0] == payoffs.shape[0]:
        return scores, payoffs

    if scores.shape[0] == 1:
        return np.broadcast_to(scores, (payoffs.shape[0], scores.shape[1])), payoffs
    if payoffs.shape[0] == 1:
        return scores, np.broadcast_to(payoffs, (scores.shape[0], payoffs.shape[1]))

    raise ValueError(
        "Scores and payoffs must have the same batch size or be broadcastable from one path. "
        f"Got scores batch={scores.shape[0]} and payoffs batch={payoffs.shape[0]}."
    )

# Validate threshold k and smoothing parameter mu
def _validate_threshold_and_mu(
    threshold: float,
    mu: float,
) -> tuple[float, float]:
    resolved_threshold = float(threshold)
    resolved_mu = float(mu)

    if not np.isfinite(resolved_threshold):
        raise ValueError(f"Threshold must be finite. Got: {threshold}")
    if resolved_threshold <= 0.0:
        raise ValueError(f"Threshold must be strictly positive. Got: {threshold}")
    if not np.isfinite(resolved_mu):
        raise ValueError(f"mu must be finite. Got: {mu}")
    if resolved_mu <= 0.0:
        raise ValueError(f"mu must be strictly positive. Got: {mu}")

    return resolved_threshold, resolved_mu

# Compute numerically stable sigmoid transformation
def _stable_sigmoid(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype = np.float64)
    output = np.empty_like(x, dtype = np.float64)

    positive = x >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

    negative_values = np.exp(x[~positive])
    output[~positive] = negative_values / (1.0 + negative_values)

    return output

# Compute smoothed stop probabilities from cumulative squared scores
def sigmoid_stop_indicator(
    cumulative_squared_scores: np.ndarray | list[float] | list[list[float]],
    *,
    threshold: float,
    mu: float,
) -> np.ndarray:
    resolved_threshold, resolved_mu = _validate_threshold_and_mu(threshold, mu)
    trace = np.asarray(cumulative_squared_scores, dtype = np.float64)

    if trace.ndim not in {1, 2}:
        raise ValueError(
            "Cumulative squared scores must have shape (P,) or (B, P). "
            f"Got shape: {trace.shape}"
        )
    if not np.all(np.isfinite(trace)):
        raise ValueError("Cumulative squared scores contain NaN or infinite value(s).")

    logits = resolved_mu * (trace - resolved_threshold)
    return _stable_sigmoid(logits)

# Compute smoothed survival probabilities from cumulative squared scores
def sigmoid_survival_indicator(
    cumulative_squared_scores: np.ndarray | list[float] | list[list[float]],
    *,
    threshold: float,
    mu: float,
) -> np.ndarray:
    stop_indicator = sigmoid_stop_indicator(
        cumulative_squared_scores,
        threshold = threshold,
        mu = mu,
    )
    return 1.0 - stop_indicator

# Build the full smoothed stopping-loss result from scores and payoffs
def build_smoothed_loss_result(
    scores: np.ndarray | list[float] | list[list[float]],
    payoffs: np.ndarray | list[float] | list[list[float]],
    *,
    threshold: float,
    mu: float,
) -> SmoothedLossResult:
    batched_scores, scores_squeezed = _coerce_batched_array(scores, name = "scores")
    batched_payoffs, payoffs_squeezed = _coerce_batched_array(payoffs, name = "payoffs")
    batched_scores, batched_payoffs = _broadcast_batch_dimension(
        batched_scores,
        batched_payoffs,
    )

    if batched_payoffs.shape[1] != batched_scores.shape[1] + 1:
        raise ValueError(
            "Payoffs must be aligned so that len(payoffs) = len(scores) + 1 along the time axis. "
            f"Got scores shape={batched_scores.shape} and payoffs shape={batched_payoffs.shape}."
        )

    cumulative_scores = compute_batched_cumulative_squared_scores(batched_scores)
    stop_indicator = sigmoid_stop_indicator(
        cumulative_scores,
        threshold = threshold,
        mu = mu,
    )
    survival_indicator = 1.0 - stop_indicator
    payoff_increments = compute_payoff_increments(batched_payoffs)

    path_values = batched_payoffs[:, 0] + np.sum(
        survival_indicator * payoff_increments,
        axis = 1,
        dtype = np.float64,
    )
    loss = -float(np.mean(path_values, dtype = np.float64))

    if scores_squeezed and payoffs_squeezed:
        return SmoothedLossResult(
            scores = batched_scores[0],
            cumulative_squared_scores = cumulative_scores[0],
            stop_indicator = stop_indicator[0],
            survival_indicator = survival_indicator[0],
            payoff_increments = payoff_increments[0],
            path_values = path_values,
            loss = loss,
        )

    return SmoothedLossResult(
        scores = batched_scores,
        cumulative_squared_scores = cumulative_scores,
        stop_indicator = stop_indicator,
        survival_indicator = survival_indicator,
        payoff_increments = payoff_increments,
        path_values = path_values,
        loss = loss,
    )

# Compute the scalar smoothed stopping loss
def smoothed_stopping_loss(
    scores: np.ndarray | list[float] | list[list[float]],
    payoffs: np.ndarray | list[float] | list[list[float]],
    *,
    threshold: float,
    mu: float,
) -> float:
    result = build_smoothed_loss_result(
        scores,
        payoffs,
        threshold = threshold,
        mu = mu,
    )
    return result.loss
