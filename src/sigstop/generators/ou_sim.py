from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from .types import OUGeneratorParams, OUSampleBatch, OUSampleRequest

SUPPORTED_SAMPLE_DEVICES = {
    "cpu",
}

# Find x0 from input spread artifact
def resolve_x0_from_spread_artifact(
    spread_path: str | Path,
    *,
    spread_column: str = "spread",
    row_index: int = 0,
) -> float:
    spread_df = pd.read_parquet(spread_path)

    if spread_column not in spread_df.columns:
        raise ValueError(
            f"Expected spread column '{spread_column}' in artifact: {spread_path}"
        )

    if row_index < 0 or row_index >= len(spread_df):
        raise IndexError(
            f"Requested row_index={row_index} outside spread artifact bounds: {len(spread_df)}"
        )

    x0 = float(spread_df.iloc[row_index][spread_column])
    if not np.isfinite(x0):
        raise ValueError(f"Resolved x0 is not finite. Got: {x0}")

    return x0

# Generate sample OU paths
def sample_ou_paths(
    params: OUGeneratorParams,
    request: OUSampleRequest,
) -> OUSampleBatch:
    _validate_sample_request(request)
    _validate_generator_params(params)

    device = str(request.device).lower()
    if device not in SUPPORTED_SAMPLE_DEVICES:
        raise ValueError(
            f"Unsupported sample device: {request.device}. "
            f"Supported devices: {sorted(SUPPORTED_SAMPLE_DEVICES)}"
        )

    output_dtype = np.dtype(request.dtype)
    working_dtype = np.float64
    rng = np.random.default_rng(request.seed)

    phi = float(np.exp(-params.kappa * request.dt))
    step_variance = compute_ou_step_variance(
        kappa = params.kappa,
        sigma = params.sigma,
        dt = request.dt,
    )
    step_std = float(np.sqrt(step_variance))

    paths = np.empty((request.n_paths, request.horizon + 1), dtype = working_dtype)
    paths[:, 0] = float(request.x0)

    innovations = None
    if request.include_innovations:
        innovations = np.empty((request.n_paths, request.horizon), dtype = working_dtype)

    for step in range(request.horizon):
        z = rng.standard_normal(request.n_paths)
        current = paths[:, step]
        mean_next = compute_ou_step_mean(current, theta = params.theta, kappa = params.kappa, dt = request.dt)
        paths[:, step + 1] = mean_next + step_std * z

        if innovations is not None:
            innovations[:, step] = z

    metadata = {
        "generator_type": "ou",
        "sampling_method": "exact_transition",
        "request": request.to_dict(),
        "params": params.to_dict(),
        "phi": phi,
        "step_variance": step_variance,
        "rng": {
            "library": "numpy",
            "bit_generator": "PCG64",
            "seed": int(request.seed),
        },
        "paths_shape": [int(request.n_paths), int(request.horizon + 1)],
        "dtype": output_dtype.name,
        "device": device,
    }

    return OUSampleBatch(
        paths = paths.astype(output_dtype, copy = False),
        metadata = metadata,
        innovations = (
            None
            if innovations is None
            else innovations.astype(output_dtype, copy = False)
        ),
    )

# Compute OU step mean
def compute_ou_step_mean(
    x_t: np.ndarray,
    *,
    theta: float,
    kappa: float,
    dt: float,
) -> np.ndarray:
    phi = np.exp(-kappa * dt)
    return theta + (x_t - theta) * phi

# Compute OU step variance (std)
def compute_ou_step_variance(
    *,
    kappa: float,
    sigma: float,
    dt: float,
) -> float:
    reversion_term = -np.expm1(-2.0 * kappa * dt)
    variance = (sigma ** 2) * reversion_term / (2.0 * kappa)

    if variance <= 0.0 or not np.isfinite(variance):
        raise ValueError(f"OU step variance must be positive and finite. Got: {variance}")

    return float(variance)

# Validate sample request
def _validate_sample_request(request: OUSampleRequest) -> None:
    if request.n_paths <= 0:
        raise ValueError(f"n_paths must be positive. Got: {request.n_paths}")
    if request.horizon < 0:
        raise ValueError(f"horizon must be nonnegative. Got: {request.horizon}")
    if request.dt <= 0:
        raise ValueError(f"dt must be positive. Got: {request.dt}")
    if not np.isfinite(request.x0):
        raise ValueError(f"x0 must be finite. Got: {request.x0}")

# Validate generator parameters
def _validate_generator_params(params: OUGeneratorParams) -> None:
    if params.kappa <= 0:
        raise ValueError(f"kappa must be positive. Got: {params.kappa}")
    if params.sigma <= 0:
        raise ValueError(f"sigma must be positive. Got: {params.sigma}")
    if params.dt <= 0:
        raise ValueError(f"params.dt must be positive. Got: {params.dt}")
