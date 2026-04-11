from __future__ import annotations
from dataclasses import asdict
from typing import Any
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, kpss
from src.sigstop.spread.ou import OUFitResult, validate_spread_series

# Run Augmented Dickey-Fuller test
# Rule to pass: p value < alpha
def run_adf_test(spread: pd.Series | np.ndarray, alpha: float = 0.05, regression: str = "c", autolag: str = "AIC") -> dict[str, Any]:
    x = validate_spread_series(spread)
    stat, p_value, usedlag, nobs, crit_vals, icbest = adfuller(x, regression=regression, autolag=autolag)

    return {
        "test": "ADF",
        "statistic": float(stat),
        "p_value": float(p_value),
        "used_lag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit_vals.items()},
        "icbest": float(icbest) if icbest is not None else None,
        "regression": regression,
        "autolag": autolag,
        "pass_threshold": alpha,
        "pass": bool(p_value < alpha),
    }

# Run KPSS test
# Rule to pass: p value > alpha
def run_kpss_test(spread: pd.Series | np.ndarray, alpha: float = 0.05, regression: str = "c", nlags: str | int = "auto") -> dict[str, Any]:
    x = validate_spread_series(spread)

    stat, p_value, lags, crit_vals = kpss(
        x,
        regression=regression,
        nlags=nlags,
    )

    return {
        "test": "KPSS",
        "statistic": float(stat),
        "p_value": float(p_value),
        "used_lag": int(lags),
        "critical_values": {k: float(v) for k, v in crit_vals.items()},
        "regression": regression,
        "nlags": nlags,
        "pass_threshold": alpha,
        "pass": bool(p_value > alpha),
    }

# Build half-life summary from fitted OU parameters
def compute_half_life_diagnostic(ou_fit: OUFitResult, warn_range: tuple[float, float] = (3.0, 90.0)) -> dict[str, Any]:
    half_life = float(ou_fit.half_life_days)
    lo, hi = warn_range

    if half_life < lo:
        interpretation = "too_fast"
    elif half_life > hi:
        interpretation = "too_slow"
    else:
        interpretation = "acceptable"

    return {
        "alpha": float(ou_fit.alpha),
        "half_life_days": half_life,
        "warn_range": [float(lo), float(hi)],
        "interpretation": interpretation,
        "pass": bool(lo <= half_life <= hi),
    }

# Compute ACF values
def compute_acf_diagnostic(spread: pd.Series | np.ndarray, nlags: int = 60, fft: bool = True) -> dict[str, Any]:
    x = validate_spread_series(spread)

    acf_vals = acf(
        x,
        nlags=nlags,
        fft=fft,
    )

    # acf returns lag 0..nlags --> skip lag 0 in stored series
    lags = list(range(1, nlags + 1))
    values = [float(v) for v in acf_vals[1:]]

    return {
        "nlags": int(nlags),
        "fft": bool(fft),
        "lags": lags,
        "values": values,
    }

# Run all diagnostic functions and return single summary dictionary
def build_spread_diagnostics_summary(spread: pd.Series | np.ndarray, ou_fit: OUFitResult, adf_alpha: float = 0.05, kpss_alpha: float = 0.05, half_life_warn_range: tuple[float, float] = (3.0, 90.0), acf_nlags: int = 60) -> dict[str, Any]:
    x = validate_spread_series(spread)

    adf_result = run_adf_test(
        spread=x,
        alpha=adf_alpha,
        regression="c",
        autolag="AIC",
    )

    kpss_result = run_kpss_test(
        spread=x,
        alpha=kpss_alpha,
        regression="c",
        nlags="auto",
    )

    half_life_result = compute_half_life_diagnostic(
        ou_fit=ou_fit,
        warn_range=half_life_warn_range,
    )

    acf_result = compute_acf_diagnostic(
        spread=x,
        nlags=acf_nlags,
        fft=True,
    )

    overall = {
        "adf_pass": adf_result["pass"],
        "kpss_pass": kpss_result["pass"],
        "half_life_pass": half_life_result["pass"],
    }

    return {
        "ou_fit": {
            "alpha": float(ou_fit.alpha),
            "theta": float(ou_fit.theta),
            "sigma": float(ou_fit.sigma),
            "log_likelihood": float(ou_fit.log_likelihood),
            "neg_log_likelihood": float(ou_fit.neg_log_likelihood),
            "half_life_days": float(ou_fit.half_life_days),
            "n_obs": int(ou_fit.n_obs),
            "success": bool(ou_fit.success),
            "message": str(ou_fit.message),
        },
        "adf": adf_result,
        "kpss": kpss_result,
        "half_life": half_life_result,
        "acf": acf_result,
        "overall": overall,
    }