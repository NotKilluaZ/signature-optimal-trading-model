from __future__ import annotations
import math
import numpy as np
from .types import OUCurveErrorSummary, OUGeneratorParams, OUSampleRequest, OUSampleValidationReport

# Compute various theoretical OU values and compare to our generated OU results to ensure valid yields
def build_ou_validation_report(
    paths: np.ndarray,
    *,
    params: OUGeneratorParams,
    request: OUSampleRequest,
    reference_spread: np.ndarray | None = None,
    acf_nlags: int = 60,
) -> OUSampleValidationReport:
    base_report = validate_sample_batch(
        paths,
        expected_n_paths = request.n_paths,
        expected_horizon = request.horizon,
        expected_x0 = request.x0,
    )

    theoretical_mean = compute_theoretical_mean_curve(
        x0 = request.x0,
        theta = params.theta,
        kappa = params.kappa,
        dt = request.dt,
        horizon = request.horizon,
    )
    theoretical_variance = compute_theoretical_variance_curve(
        kappa = params.kappa,
        sigma = params.sigma,
        dt = request.dt,
        horizon = request.horizon,
    )

    empirical_mean = np.mean(paths, axis = 0)
    empirical_variance = np.var(paths, axis = 0)
    mean_curve_error = summarize_curve_error(empirical_mean, theoretical_mean)
    variance_curve_error = summarize_curve_error(empirical_variance, theoretical_variance)

    normalized_innovations = compute_normalized_innovations(
        paths,
        theta = params.theta,
        kappa = params.kappa,
        sigma = params.sigma,
        dt = request.dt,
    )
    innovation_summary = summarize_innovations(normalized_innovations)

    acf_summary = compute_acf_summary(
        paths,
        reference_spread = reference_spread,
        nlags = acf_nlags,
    )
    empirical_half_life_summary = compute_empirical_half_life_summary(
        paths,
        theta = params.theta,
        theoretical_half_life = params.half_life_days,
    )

    checks = dict(base_report.checks)
    checks["step_variance_positive"] = bool(theoretical_variance[-1] > 0.0)
    checks["innovation_mean_reasonable"] = bool(abs(innovation_summary["mean"]) <= 0.05)
    checks["innovation_variance_reasonable"] = bool(
        abs(innovation_summary["variance"] - 1.0) <= 0.15
    )
    checks["mean_curve_reasonable"] = _curve_error_within(mean_curve_error, threshold = 0.15)
    checks["variance_curve_reasonable"] = _curve_error_within(
        variance_curve_error,
        threshold = 0.20,
    )

    notes = list(base_report.notes)
    notes.append(
        "Theoretical mean/variance checks compare simulated ensemble moments to exact OU transition formulas."
    )
    notes.append(
        "Innovation diagnostics normalize one-step residuals by the exact OU transition standard deviation."
    )
    if reference_spread is None:
        notes.append("ACF summary excludes a real-spread comparison because no reference spread was provided.")
    else:
        notes.append("ACF summary compares average simulated-path autocorrelation to the formation spread.")

    status = "pass" if all(checks.values()) else "fail"

    return OUSampleValidationReport(
        status = status,
        checks = checks,
        notes = notes,
        innovation_summary = innovation_summary,
        mean_curve_error = mean_curve_error,
        variance_curve_error = variance_curve_error,
        acf_summary = acf_summary,
        empirical_half_life_summary = empirical_half_life_summary,
    )


def validate_sample_batch(
    paths: np.ndarray,
    *,
    expected_n_paths: int,
    expected_horizon: int,
    expected_x0: float | None = None,
) -> OUSampleValidationReport:
    checks = {
        "finite": bool(np.all(np.isfinite(paths))),
        "shape": bool(paths.shape == (expected_n_paths, expected_horizon + 1)),
    }

    if expected_x0 is not None:
        checks["x0_column"] = bool(np.allclose(paths[:, 0], expected_x0))

    status = "pass" if all(checks.values()) else "fail"

    return OUSampleValidationReport(
        status = status,
        checks = checks,
        notes = [
            "Step 5 validation checks finite values, shape, and x0 alignment before richer diagnostics.",
        ],
    )


def compute_theoretical_mean_curve(
    *,
    x0: float,
    theta: float,
    kappa: float,
    dt: float,
    horizon: int,
) -> np.ndarray:
    steps = np.arange(horizon + 1, dtype = float)
    return theta + (x0 - theta) * np.exp(-kappa * dt * steps)


def compute_theoretical_variance_curve(
    *,
    kappa: float,
    sigma: float,
    dt: float,
    horizon: int,
) -> np.ndarray:
    steps = np.arange(horizon + 1, dtype = float)
    variance = (sigma ** 2) * (-np.expm1(-2.0 * kappa * dt * steps)) / (2.0 * kappa)
    variance[0] = 0.0
    return variance


def summarize_curve_error(empirical: np.ndarray, theoretical: np.ndarray) -> OUCurveErrorSummary:
    if empirical.shape != theoretical.shape:
        raise ValueError(
            f"Curve shape mismatch. empirical={empirical.shape}, theoretical={theoretical.shape}"
        )

    abs_error = np.abs(empirical - theoretical)
    rel_mask = np.abs(theoretical) > 1e-8
    rel_error = np.full_like(abs_error, np.nan, dtype = float)
    rel_error[rel_mask] = abs_error[rel_mask] / np.abs(theoretical[rel_mask])

    return OUCurveErrorSummary(
        max_abs_error = float(np.max(abs_error)),
        mean_abs_error = float(np.mean(abs_error)),
        max_rel_error = _nanmax_or_none(rel_error),
        mean_rel_error = _nanmean_or_none(rel_error),
    )


def compute_normalized_innovations(
    paths: np.ndarray,
    *,
    theta: float,
    kappa: float,
    sigma: float,
    dt: float,
) -> np.ndarray:
    x_t = paths[:, :-1]
    x_next = paths[:, 1:]

    mean_next = theta + (x_t - theta) * np.exp(-kappa * dt)
    step_variance = (sigma ** 2) * (-np.expm1(-2.0 * kappa * dt)) / (2.0 * kappa)
    step_std = math.sqrt(step_variance)

    return (x_next - mean_next) / step_std


def summarize_innovations(normalized_innovations: np.ndarray) -> dict[str, float]:
    flat = normalized_innovations.reshape(-1)
    quantiles = np.quantile(flat, [0.01, 0.05, 0.50, 0.95, 0.99])

    return {
        "count": int(flat.size),
        "mean": float(np.mean(flat)),
        "variance": float(np.var(flat)),
        "std": float(np.std(flat)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q95": float(quantiles[3]),
        "q99": float(quantiles[4]),
    }


def compute_empirical_half_life_summary(
    paths: np.ndarray,
    *,
    theta: float,
    theoretical_half_life: float,
) -> dict[str, float | int | None]:
    initial_distance = np.abs(paths[:, 0] - theta)
    hit_threshold = 0.5 * initial_distance

    half_lives = np.full(paths.shape[0], np.nan, dtype = float)

    for path_idx in range(paths.shape[0]):
        distance = np.abs(paths[path_idx] - theta)
        if hit_threshold[path_idx] == 0.0:
            half_lives[path_idx] = 0.0
            continue

        hit_indices = np.flatnonzero(distance <= hit_threshold[path_idx])
        if hit_indices.size > 0:
            half_lives[path_idx] = float(hit_indices[0])

    valid = np.isfinite(half_lives)
    median_empirical = float(np.median(half_lives[valid])) if np.any(valid) else None

    return {
        "n_paths_hit": int(np.sum(valid)),
        "n_paths_total": int(paths.shape[0]),
        "hit_ratio": float(np.mean(valid)),
        "theoretical_half_life_days": float(theoretical_half_life),
        "median_empirical_half_life_days": median_empirical,
        "median_abs_error_days": (
            None
            if median_empirical is None
            else float(abs(median_empirical - theoretical_half_life))
        ),
    }


def compute_acf_summary(
    paths: np.ndarray,
    *,
    reference_spread: np.ndarray | None,
    nlags: int,
) -> dict[str, object]:
    max_nlags = min(nlags, paths.shape[1] - 1)
    if max_nlags <= 0:
        return {
            "nlags": 0,
            "lags": [],
            "ensemble_mean_acf": [],
            "reference_acf": None,
            "curve_error": None,
        }

    ensemble_mean_acf = compute_ensemble_acf(paths, nlags = max_nlags)
    reference_acf = None
    curve_error = None

    if reference_spread is not None:
        reference_acf = compute_acf_1d(reference_spread, nlags = max_nlags)
        curve_error = summarize_curve_error(ensemble_mean_acf, reference_acf).to_dict()

    return {
        "nlags": int(max_nlags),
        "lags": list(range(1, max_nlags + 1)),
        "ensemble_mean_acf": [float(v) for v in ensemble_mean_acf],
        "reference_acf": None if reference_acf is None else [float(v) for v in reference_acf],
        "curve_error": curve_error,
    }


def compute_ensemble_acf(paths: np.ndarray, *, nlags: int) -> np.ndarray:
    centered = paths - np.mean(paths, axis = 1, keepdims = True)
    denom = np.sum(centered ** 2, axis = 1)
    valid = denom > 0.0
    if not np.any(valid):
        return np.zeros(nlags, dtype = float)

    acf_values = np.zeros(nlags, dtype = float)
    centered_valid = centered[valid]
    denom_valid = denom[valid]

    for lag in range(1, nlags + 1):
        numerator = np.sum(centered_valid[:, :-lag] * centered_valid[:, lag:], axis = 1)
        acf_values[lag - 1] = float(np.mean(numerator / denom_valid))

    return acf_values


def compute_acf_1d(series: np.ndarray, *, nlags: int) -> np.ndarray:
    x = np.asarray(series, dtype = float)
    if x.ndim != 1:
        raise ValueError(f"Expected 1D series for ACF computation. Got: {x.shape}")
    if len(x) <= nlags:
        raise ValueError(f"Series length must exceed nlags. Got length={len(x)}, nlags={nlags}")

    centered = x - np.mean(x)
    denom = float(np.dot(centered, centered))
    if denom <= 0.0:
        return np.zeros(nlags, dtype = float)

    acf_values = np.zeros(nlags, dtype = float)
    for lag in range(1, nlags + 1):
        acf_values[lag - 1] = float(np.dot(centered[:-lag], centered[lag:]) / denom)
    return acf_values


def _curve_error_within(summary: OUCurveErrorSummary, *, threshold: float) -> bool:
    if summary.mean_rel_error is None:
        return True
    return bool(summary.mean_rel_error <= threshold)


def _nanmean_or_none(values: np.ndarray) -> float | None:
    if np.all(np.isnan(values)):
        return None
    return float(np.nanmean(values))


def _nanmax_or_none(values: np.ndarray) -> float | None:
    if np.all(np.isnan(values)):
        return None
    return float(np.nanmax(values))
