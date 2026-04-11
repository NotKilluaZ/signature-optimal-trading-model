from __future__ import annotations
import numpy as np

# Validate input spread paths shape and values
def _validate_spread_paths(
    spread_paths: np.ndarray | list[float] | list[list[float]],
) -> tuple[np.ndarray, bool]:
    x = np.asarray(spread_paths, dtype = np.float64)
    squeezed = False

    if x.ndim == 1:
        x = x[None, :]
        squeezed = True
    elif x.ndim != 2:
        raise ValueError(
            "Spread paths must have shape (L,) or (B, L). "
            f"Got shape: {x.shape}"
        )

    if x.shape[1] == 0:
        raise ValueError("Spread paths must contain at least one observation.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Spread paths contain NaN or infinite value(s).")

    return x, squeezed

# Build & validate observations time grid
def _build_time_grid(
    length: int,
    *,
    dt: float = 1.0,
    times: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    if length <= 0:
        raise ValueError(f"Length must be positive. Got: {length}")

    if times is not None:
        resolved_times = np.asarray(times, dtype = np.float64)
        if resolved_times.ndim != 1 or resolved_times.shape[0] != length:
            raise ValueError(
                "Explicit times must be a 1D array aligned to the spread path length. "
                f"Expected length={length}, got shape={resolved_times.shape}"
            )
        if not np.all(np.isfinite(resolved_times)):
            raise ValueError("Explicit times contain NaN or infinite value(s).")
        if np.any(np.diff(resolved_times) < 0.0):
            raise ValueError("Explicit times must be non-decreasing.")
        return resolved_times

    resolved_dt = float(dt)
    if not np.isfinite(resolved_dt):
        raise ValueError(f"dt must be finite. Got: {dt}")
    if resolved_dt <= 0.0:
        raise ValueError(f"dt must be strictly positive. Got: {dt}")

    return np.arange(length, dtype = np.float64) * resolved_dt

# Calculate discount factors
def compute_discount_factors(
    length: int,
    *,
    rate: float = 0.0,
    dt: float = 1.0,
    times: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    resolved_rate = float(rate)
    if not np.isfinite(resolved_rate):
        raise ValueError(f"Rate must be finite. Got: {rate}")

    time_grid = _build_time_grid(length, dt = dt, times = times)
    return np.exp(-resolved_rate * time_grid, dtype = np.float64)

# Compute discounted payoff from entering the spread position at each time
def build_entry_payoff_process(
    spread_paths: np.ndarray | list[float] | list[list[float]],
    *,
    r: float = 0.0,
    c: float = 0.0,
    dt: float = 1.0,
    times: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    x, squeezed = _validate_spread_paths(spread_paths)
    resolved_cost = float(c)

    if not np.isfinite(resolved_cost):
        raise ValueError(f"Transaction cost c must be finite. Got: {c}")

    discounts = compute_discount_factors(
        x.shape[1],
        rate = r,
        dt = dt,
        times = times,
    )[None, :]
    payoff = discounts * (-x - resolved_cost)

    return payoff[0] if squeezed else payoff

# Compute discounted payoff from exiting the spread position at each time
def build_exit_payoff_process(
    spread_paths: np.ndarray | list[float] | list[list[float]],
    *,
    r_hat: float = 0.0,
    c_hat: float = 0.0,
    dt: float = 1.0,
    times: np.ndarray | list[float] | None = None,
) -> np.ndarray:
    x, squeezed = _validate_spread_paths(spread_paths)
    resolved_cost = float(c_hat)

    if not np.isfinite(resolved_cost):
        raise ValueError(f"Transaction cost c_hat must be finite. Got: {c_hat}")

    discounts = compute_discount_factors(
        x.shape[1],
        rate = r_hat,
        dt = dt,
        times = times,
    )[None, :]
    payoff = discounts * (x - resolved_cost)

    return payoff[0] if squeezed else payoff

# Compute one-step increments of a payoff process
def compute_payoff_increments(
    payoff_process: np.ndarray | list[float] | list[list[float]]
) -> np.ndarray:
    x, squeezed = _validate_spread_paths(payoff_process)

    if x.shape[1] < 2:
        raise ValueError(
            "Payoff process must contain at least two observations to compute increments."
        )

    increments = np.diff(x, axis = 1)
    return increments[0] if squeezed else increments
