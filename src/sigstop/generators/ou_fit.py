from __future__ import annotations
import math
from pathlib import Path
import pandas as pd
from src.sigstop.spread.ou import OUFitResult, fit_ou_mle, validate_spread_series
from .types import OUFitDiagnostics, OUGeneratorParams

SUPPORTED_OU_FIT_METHODS = {
    "transition_mle",
}

# Fit OU parameters onto spread series using transition density MLE
def fit_ou_generator_params(
    spread: pd.Series,
    *,
    dt: float = 1.0,
    var_floor: float = 1e-12,
    method: str = "transition_mle",
) -> tuple[OUGeneratorParams, OUFitDiagnostics]:
    resolved_method = str(method)
    if resolved_method not in SUPPORTED_OU_FIT_METHODS:
        raise ValueError(
            f"Unsupported OU generator fit method: {resolved_method}. "
            f"Supported methods: {sorted(SUPPORTED_OU_FIT_METHODS)}"
        )

    x = validate_spread_series(spread)
    fit_result = fit_ou_mle(x, dt = dt, var_floor = var_floor)

    params = convert_spread_ou_fit_to_generator_params(
        fit_result,
        dt = dt,
        estimation_method = resolved_method,
    )
    diagnostics = build_ou_fit_diagnostics(
        fit_result,
        var_floor = var_floor,
        optimizer_type = "L-BFGS-B",
    )

    return params, diagnostics

# Load spread series from directory and validate before fitting OU params onto it
def fit_ou_generator_params_from_artifact(
    spread_path: str | Path,
    *,
    spread_column: str = "spread",
    dt: float = 1.0,
    var_floor: float = 1e-12,
    method: str = "transition_mle",
) -> tuple[OUGeneratorParams, OUFitDiagnostics]:
    spread_df = pd.read_parquet(spread_path)

    if spread_column not in spread_df.columns:
        raise ValueError(
            f"Expected spread column '{spread_column}' in artifact: {spread_path}"
        )

    return fit_ou_generator_params(
        spread_df[spread_column],
        dt = dt,
        var_floor = var_floor,
        method = method,
    )

# Convert spread OU fit results into new OUGeneratorParams format (what the syntehtic ou path generator expects)
def convert_spread_ou_fit_to_generator_params(
    fit_result: OUFitResult,
    *,
    dt: float,
    estimation_method: str,
) -> OUGeneratorParams:
    kappa = float(fit_result.alpha)
    phi = float(math.exp(-kappa * dt))

    return OUGeneratorParams(
        kappa = kappa,
        theta = float(fit_result.theta),
        sigma = float(fit_result.sigma),
        dt = float(dt),
        phi = phi,
        half_life_days = float(fit_result.half_life_days),
        estimation_method = estimation_method,
    )

# Create diagnostic logs
def build_ou_fit_diagnostics(
    fit_result: OUFitResult,
    *,
    var_floor: float,
    optimizer_type: str | None = None,
    fit_window: str = "formation",
) -> OUFitDiagnostics:
    return OUFitDiagnostics(
        log_likelihood = float(fit_result.log_likelihood),
        neg_log_likelihood = float(fit_result.neg_log_likelihood),
        n_obs = int(fit_result.n_obs),
        success = bool(fit_result.success),
        message = str(fit_result.message),
        var_floor = float(var_floor),
        fit_window = fit_window,
        optimizer_type = optimizer_type,
    )
