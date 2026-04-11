from __future__ import annotations
import numpy as np
import pandas as pd

# Resolve data type
def _resolve_dtype(dtype: str | np.dtype | type[np.floating]) -> np.dtype:
    resolved = np.dtype(dtype)
    if resolved not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"Unsupported dtype for augmentation: {resolved}")
    return resolved

# Validate spread series is a clean 1D float array
def validate_spread_array(spread: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(spread, dtype = float)

    if x.ndim != 1:
        raise ValueError(f"Spread must be 1 dimensional. Got shape: {x.shape}")
    if len(x) < 2:
        raise ValueError("Spread must contain at least 2 observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("Spread series contains NaN or infinite value(s)")

    return x

# Validate batched spread array shape and size
def validate_batched_spread_array(spread: np.ndarray | list[list[float]]) -> np.ndarray:
    x = np.asarray(spread, dtype = float)

    if x.ndim != 2:
        raise ValueError(
            f"Batched spread input must be 2 dimensional. Got shape: {x.shape}"
        )
    if x.shape[0] < 1:
        raise ValueError("Batched spread input must contain at least one path.")
    if x.shape[1] < 2:
        raise ValueError("Each spread path must contain at least 2 observations.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Batched spread input contains NaN or infinite value(s)")

    return x

# Create deterministic time channel with a frozen representation rule
def make_time_channel(
    length: int,
    representation: str = "index_0_1",
    dtype: str | np.dtype | type[np.floating] = np.float32,
) -> np.ndarray:
    if length < 2:
        raise ValueError(f"Input length for time channel less than 2. Got: {length}")

    resolved_dtype = _resolve_dtype(dtype)

    if representation == "index_0_1":
        time_channel = np.arange(length, dtype = float) / (length - 1)
    elif representation == "index_int":
        time_channel = np.arange(length, dtype = float)
    else:
        raise ValueError(
            "Unsupported time representation. "
            f"Expected one of ['index_0_1', 'index_int'], got: {representation}"
        )

    return time_channel.astype(resolved_dtype, copy = False)

# Convert 1D spread series into 2D augmented path
def augment_spread_with_time(
    spread: pd.Series | np.ndarray,
    dtype: str | np.dtype | type[np.floating] = np.float32,
    time_representation: str = "index_0_1",
) -> np.ndarray:
    resolved_dtype = _resolve_dtype(dtype)
    x = validate_spread_array(spread).astype(resolved_dtype, copy = False)
    t = make_time_channel(
        len(x),
        representation = time_representation,
        dtype = resolved_dtype,
    )

    augmented = np.column_stack([t, x]).astype(resolved_dtype, copy = False)

    if augmented.shape != (len(x), 2):
        raise ValueError(
            f"Augmented path has wrong shape: {augmented.shape}. "
            f"Expected: ({len(x)}, 2)"
        )

    if not np.all(np.isfinite(augmented)):
        raise ValueError("Augmented path has NaN or infinite value(s)")

    return augmented

# Augment batched spreads with time
def augment_batched_spreads_with_time(
    spread: np.ndarray | list[list[float]],
    dtype: str | np.dtype | type[np.floating] = np.float32,
    time_representation: str = "index_0_1",
) -> np.ndarray:
    resolved_dtype = _resolve_dtype(dtype)
    x = validate_batched_spread_array(spread).astype(resolved_dtype, copy = False)
    t = make_time_channel(
        x.shape[1],
        representation = time_representation,
        dtype = resolved_dtype,
    )

    augmented = np.empty((x.shape[0], x.shape[1], 2), dtype = resolved_dtype)
    augmented[:, :, 0] = t[None, :]
    augmented[:, :, 1] = x

    if augmented.shape != (x.shape[0], x.shape[1], 2):
        raise ValueError(
            f"Augmented batched path has wrong shape: {augmented.shape}. "
            f"Expected: ({x.shape[0]}, {x.shape[1]}, 2)"
        )

    if not np.all(np.isfinite(augmented)):
        raise ValueError("Augmented batched path has NaN or infinite value(s)")

    return augmented
