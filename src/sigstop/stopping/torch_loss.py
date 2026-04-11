from __future__ import annotations
from dataclasses import dataclass
import torch


@dataclass(frozen = True)
class TorchSmoothedLossResult:
    scores: torch.Tensor
    cumulative_squared_scores: torch.Tensor
    stop_indicator: torch.Tensor
    survival_indicator: torch.Tensor
    payoff_increments: torch.Tensor
    path_values: torch.Tensor
    loss: torch.Tensor


# Convert input values into a validated batched 2D torch tensor
def _coerce_batched_tensor(
    values: torch.Tensor,
    *,
    name: str,
) -> tuple[torch.Tensor, bool]:
    squeezed = False
    x = values

    if x.ndim == 1:
        x = x.unsqueeze(0)
        squeezed = True
    elif x.ndim != 2:
        raise ValueError(
            f"{name} must have shape (T,) or (B, T). Got shape: {tuple(x.shape)}"
        )

    if x.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one value.")
    if not torch.isfinite(x).all():
        raise ValueError(f"{name} contains NaN or infinite value(s).")

    return x, squeezed


# Match score and payoff batch dimensions for paired torch computation
def _broadcast_batch_dimension(
    scores: torch.Tensor,
    payoffs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if scores.shape[0] == payoffs.shape[0]:
        return scores, payoffs

    if scores.shape[0] == 1:
        return scores.expand(payoffs.shape[0], -1), payoffs
    if payoffs.shape[0] == 1:
        return scores, payoffs.expand(scores.shape[0], -1)

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

    if not torch.isfinite(torch.tensor(resolved_threshold)):
        raise ValueError(f"Threshold must be finite. Got: {threshold}")
    if resolved_threshold <= 0.0:
        raise ValueError(f"Threshold must be strictly positive. Got: {threshold}")
    if not torch.isfinite(torch.tensor(resolved_mu)):
        raise ValueError(f"mu must be finite. Got: {mu}")
    if resolved_mu <= 0.0:
        raise ValueError(f"mu must be strictly positive. Got: {mu}")

    return resolved_threshold, resolved_mu


# Compute smoothed stop probabilities from cumulative squared scores
def torch_sigmoid_stop_indicator(
    cumulative_squared_scores: torch.Tensor,
    *,
    threshold: float,
    mu: float,
) -> torch.Tensor:
    resolved_threshold, resolved_mu = _validate_threshold_and_mu(threshold, mu)

    if cumulative_squared_scores.ndim not in {1, 2}:
        raise ValueError(
            "Cumulative squared scores must have shape (P,) or (B, P). "
            f"Got shape: {tuple(cumulative_squared_scores.shape)}"
        )
    if not torch.isfinite(cumulative_squared_scores).all():
        raise ValueError("Cumulative squared scores contain NaN or infinite value(s).")

    logits = resolved_mu * (cumulative_squared_scores - resolved_threshold)
    return torch.sigmoid(logits)


# Compute smoothed survival probabilities from cumulative squared scores
def torch_sigmoid_survival_indicator(
    cumulative_squared_scores: torch.Tensor,
    *,
    threshold: float,
    mu: float,
) -> torch.Tensor:
    return 1.0 - torch_sigmoid_stop_indicator(
        cumulative_squared_scores,
        threshold = threshold,
        mu = mu,
    )


# Build the full torch smoothed stopping-loss result from scores and payoffs
def build_torch_smoothed_loss_result(
    scores: torch.Tensor,
    payoffs: torch.Tensor,
    *,
    threshold: float,
    mu: float,
) -> TorchSmoothedLossResult:
    batched_scores, scores_squeezed = _coerce_batched_tensor(scores, name = "scores")
    batched_payoffs, payoffs_squeezed = _coerce_batched_tensor(payoffs, name = "payoffs")
    batched_scores, batched_payoffs = _broadcast_batch_dimension(
        batched_scores,
        batched_payoffs,
    )

    if batched_payoffs.shape[1] != batched_scores.shape[1] + 1:
        raise ValueError(
            "Payoffs must be aligned so that len(payoffs) = len(scores) + 1 along the time axis. "
            f"Got scores shape={tuple(batched_scores.shape)} and payoffs shape={tuple(batched_payoffs.shape)}."
        )

    cumulative_scores = torch.cumsum(torch.square(batched_scores), dim = 1)
    stop_indicator = torch_sigmoid_stop_indicator(
        cumulative_scores,
        threshold = threshold,
        mu = mu,
    )
    survival_indicator = 1.0 - stop_indicator
    payoff_increments = torch.diff(batched_payoffs, dim = 1)
    path_values = batched_payoffs[:, 0] + torch.sum(
        survival_indicator * payoff_increments,
        dim = 1,
    )
    loss = -torch.mean(path_values)

    if scores_squeezed and payoffs_squeezed:
        return TorchSmoothedLossResult(
            scores = batched_scores[0],
            cumulative_squared_scores = cumulative_scores[0],
            stop_indicator = stop_indicator[0],
            survival_indicator = survival_indicator[0],
            payoff_increments = payoff_increments[0],
            path_values = path_values,
            loss = loss,
        )

    return TorchSmoothedLossResult(
        scores = batched_scores,
        cumulative_squared_scores = cumulative_scores,
        stop_indicator = stop_indicator,
        survival_indicator = survival_indicator,
        payoff_increments = payoff_increments,
        path_values = path_values,
        loss = loss,
    )


# Compute the scalar torch smoothed stopping loss
def torch_smoothed_stopping_loss(
    scores: torch.Tensor,
    payoffs: torch.Tensor,
    *,
    threshold: float,
    mu: float,
) -> torch.Tensor:
    return build_torch_smoothed_loss_result(
        scores,
        payoffs,
        threshold = threshold,
        mu = mu,
    ).loss
