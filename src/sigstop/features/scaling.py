from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import asdict, dataclass

@dataclass
class ZScoreScaler:
    mean_: float
    std_: float
    fitted_on: str = "formation_only"
    method: str = "zscore"

# Resolve data type
def _resolve_dtype(dtype: str | np.dtype | type[np.floating]) -> np.dtype:
    resolved = np.dtype(dtype)
    if resolved not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"Unsupported dtype for scaling: {resolved}")
    return resolved

# Validate spread array
def validate_spread_array(spread: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(spread, dtype = float)

    if x.ndim != 1:
        raise ValueError(f"Spread must be 1D. Got shape: {x.shape}")
    if len(x) < 2:
        raise ValueError("Spread must contain at least 2 observations.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Spread contains NaN or infinite values.")

    return x

# Validate batched spread array
def validate_batched_spread_array(spread: np.ndarray | list[list[float]]) -> np.ndarray:
    x = np.asarray(spread, dtype = float)

    if x.ndim != 2:
        raise ValueError(f"Batched spread array must be 2D. Got shape: {x.shape}")
    if x.shape[0] < 1:
        raise ValueError("Batched spread array must contain at least one path.")
    if x.shape[1] < 2:
        raise ValueError("Each spread path must contain at least 2 observations.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Batched spread array contains NaN or infinite values.")

    return x

# Fit z-score scaling params on formation spread
def fit_zscore_scaler(formation_spread: pd.Series | np.ndarray, ddof: int = 1, std_floor: float = 1e-8) -> ZScoreScaler:
    x = validate_spread_array(formation_spread)

    mean_ = float(np.mean(x))
    std_ = float(np.std(x, ddof = ddof))

    if not np.isfinite(mean_):
        raise ValueError("Computed mean is not finite.")
    if not np.isfinite(std_):
        raise ValueError("Computed std is not finite.")

    std_ = max(std_, std_floor)

    return ZScoreScaler(
        mean_ = mean_,
        std_ = std_,
        fitted_on = "formation_only",
        method = "zscore",
    )

# Apply fitted z-score scaler onto spread series
def apply_zscore_scaler(spread: pd.Series | np.ndarray, scaler: ZScoreScaler, dtype: str | np.dtype | type[np.floating] = np.float32) -> np.ndarray:
    resolved_dtype = _resolve_dtype(dtype)
    x = validate_spread_array(spread)

    if scaler.std_ <= 0:
        raise ValueError(f"Scaler std must be positive. Got: {scaler.std_}")

    scaled = (x - scaler.mean_) / scaler.std_

    if not np.all(np.isfinite(scaled)):
        raise ValueError("Scaled spread contains NaN or infinite values.")

    return scaled.astype(resolved_dtype, copy = False)

# Apply batched z-score scaler
def apply_batched_zscore_scaler(
    spread: np.ndarray | list[list[float]],
    scaler: ZScoreScaler,
    dtype: str | np.dtype | type[np.floating] = np.float32,
) -> np.ndarray:
    resolved_dtype = _resolve_dtype(dtype)
    x = validate_batched_spread_array(spread)

    if scaler.std_ <= 0:
        raise ValueError(f"Scaler std must be positive. Got: {scaler.std_}")

    scaled = (x - scaler.mean_) / scaler.std_

    if not np.all(np.isfinite(scaled)):
        raise ValueError("Scaled batched spread contains NaN or infinite values.")

    return scaled.astype(resolved_dtype, copy = False)

# Convert scaler to metadata
def scaler_to_dict(scaler: ZScoreScaler) -> dict[str, float | str]:
    return asdict(scaler)
