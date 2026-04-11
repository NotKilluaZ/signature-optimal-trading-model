from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen = True)
class StopRuleResult:
    scores: np.ndarray
    cumulative_squared_scores: np.ndarray
    threshold: float
    stop_index: int | None
    crossed: bool

# Validate threshold k (must be real number and positive)
def _validate_threshold(threshold: float) -> float:
    resolved = float(threshold)
    if not np.isfinite(resolved):
        raise ValueError(f"Threshold must be finite. Got: {threshold}")
    if resolved <= 0.0:
        raise ValueError(f"Threshold must be strictly positive. Got: {threshold}")
    return resolved

# Validate score vector
def validate_score_vector(scores: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(scores, dtype = np.float64)

    if x.ndim != 1:
        raise ValueError(f"Scores must be a 1D array. Got shape: {x.shape}")
    if x.size == 0:
        raise ValueError("Scores must contain at least one value.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Scores contain NaN or infinite value(s).")

    return x

# Validate score matrix
def validate_score_matrix(scores: np.ndarray | list[list[float]]) -> np.ndarray:
    x = np.asarray(scores, dtype = np.float64)

    if x.ndim != 2:
        raise ValueError(f"Scores must be a 2D array. Got shape: {x.shape}")
    if x.shape[1] == 0:
        raise ValueError("Score matrix must contain at least one prefix score.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Score matrix contains NaN or infinite value(s).")

    return x

# Compute cumulative sqaured scores up to current point in time
def compute_cumulative_squared_scores(scores: np.ndarray | list[float]) -> np.ndarray:
    x = validate_score_vector(scores)
    squared_scores = np.square(x, dtype = np.float64)
    return np.cumsum(squared_scores, dtype = np.float64)

# Calculate batched cumulative squared scores
def compute_batched_cumulative_squared_scores(
    scores: np.ndarray | list[list[float]]
) -> np.ndarray:
    x = validate_score_matrix(scores)
    squared_scores = np.square(x, dtype = np.float64)
    return np.cumsum(squared_scores, axis = 1, dtype = np.float64)

# Find when score first crosses specified threshold (k)
def find_first_threshold_crossing(
    cumulative_squared_scores: np.ndarray | list[float],
    threshold: float,
) -> int | None:
    trace = validate_score_vector(cumulative_squared_scores)
    resolved_threshold = _validate_threshold(threshold)

    crossed = trace >= resolved_threshold
    if not np.any(crossed):
        return None

    return int(np.flatnonzero(crossed)[0])

# Find batched first crossing of threshold k
def find_batched_first_threshold_crossings(
    cumulative_squared_scores: np.ndarray | list[list[float]],
    threshold: float,
    *,
    missing_value: int = -1,
) -> np.ndarray:
    traces = validate_score_matrix(cumulative_squared_scores)
    resolved_threshold = _validate_threshold(threshold)

    crossed = traces >= resolved_threshold
    crossing_indices = np.argmax(crossed, axis = 1).astype(np.int32, copy = False)
    has_crossing = np.any(crossed, axis = 1)

    output = np.full(traces.shape[0], int(missing_value), dtype = np.int32)
    output[has_crossing] = crossing_indices[has_crossing]
    return output

# Find index of when our score first crosses the threshold (k) --> Then stop when threshold is met
def apply_deterministic_stop_rule(
    scores: np.ndarray | list[float],
    threshold: float,
) -> StopRuleResult:
    resolved_scores = validate_score_vector(scores)
    cumulative_scores = compute_cumulative_squared_scores(resolved_scores)
    stop_index = find_first_threshold_crossing(cumulative_scores, threshold)

    return StopRuleResult(
        scores = resolved_scores,
        cumulative_squared_scores = cumulative_scores,
        threshold = float(threshold),
        stop_index = stop_index,
        crossed = stop_index is not None,
    )
