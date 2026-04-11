from __future__ import annotations
import math
import pandas as pd
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize

# Class is easier and cleaner to hold all of these parameter values instead of a long list
@dataclass
class OUFitResult:
    alpha: float
    theta: float
    sigma: float
    log_likelihood: float
    neg_log_likelihood: float
    half_life_days: float
    n_obs: int
    success: bool
    message: str

# Convert spread input into clean 1D numpy array then validate
def validate_spread_series(spread: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(spread, dtype = float)

    if x.ndim != 1:
        raise ValueError(f"Spread series input is not 1D. Got: {x.shape}")
    if len(x) < 2:
        raise ValueError("Spread series input must contain at least 2 observations")
    if not np.all(np.isfinite(x)):
        raise ValueError("Spread series contains NaN or infinite values")
    if np.var(x) <= 0:
        raise ValueError("Spread series must have nonzero variance")
    
    return x

# Compute OU conditional mean E[X_{t+dt} | X_t]
def ou_conditional_mean(x_t: np.ndarray, alpha: float, theta: float, dt: float) -> np.ndarray:
    exp_term = np.exp(-alpha * dt)
    return x_t * exp_term + theta * (1.0 - exp_term)

# Compute OU conditional variance Var[X_{t+dt} | X_t]
def ou_conditional_variance(alpha: float, sigma: float, dt: float, var_floor: float = 1e-12) -> float:
    var = sigma**2 * (1.0 - np.exp(-2.0 * alpha * dt)) / (2.0 * alpha)
    return max(var, var_floor)

# Compute total OU transition log-likelihood for spread series
def ou_transition_log_likelihod(spread: pd.Series | np.ndarray, alpha: float, theta: float, sigma: float, dt: float = 1.0, var_floor: float = 1e-12) -> float:
    x = validate_spread_series(spread)

    if alpha <= 0:
        raise ValueError(f"Alpha must be greater than 0. Got: {alpha}")
    if sigma <= 0:
        raise ValueError(f"Sigma must be greater than 0. Got: {sigma}")
    if dt <= 0:
        raise ValueError(f"Dt must be greater than 0. Got: {dt}")
    
    x_t = x[ : -1]
    x_next = x[1 : ]

    mean_t = ou_conditional_mean(x_t, alpha, theta, dt)
    var_t = ou_conditional_variance(alpha, sigma, dt, var_floor)

    loglik = -0.5 * np.sum(
        np.log(2.0 * np.pi * var_t) + ((x_next - mean_t) ** 2) / var_t
    )
    
    return float(loglik)

# Compute negative log likelihood in transformed paramter space (log alpha, theta, log sigma)
def ou_negative_log_likelihood_transformed(params: np.ndarray, spread: pd.Series | np.ndarray, dt: float = 1.0, var_floor: float = 1e-12) -> float:
    log_alpha, theta, log_sigma = params

    alpha = float(np.exp(log_alpha))
    sigma = float(np.exp(log_sigma))

    loglik = ou_transition_log_likelihod(spread, alpha, theta, sigma, dt, var_floor)

    return loglik * -1

# Build simple initial guess for parameters alpha, theta, sigma
def initial_ou_guess(spread: pd.Series | np.ndarray) -> np.ndarray:
    x = validate_spread_series(spread)
    
    alpha_not = 0.1
    theta_not = float(np.mean(x))
    sigma_not = float(np.std(np.diff(x), ddof = 1))
    sigma_not = max(sigma_not, 1e-6)

    return np.array([np.log(alpha_not), theta_not, np.log(sigma_not)], dtype = float)

# Compute OU half-life in same time units as data type (dt)
def compute_half_life(alpha: float) -> float:
    if alpha <= 0:
        raise ValueError(f"Alpha must be greater than 0. Got: {alpha}")
    return math.log(2.0) / alpha

# Fit OU parameters by maximizing log likelihood
def fit_ou_mle(spread: pd.Series | np.ndarray, dt: float = 1.0, var_floor: float = 1e-12) -> OUFitResult:
    x = validate_spread_series(spread)
    x_not = initial_ou_guess(x)

    result = minimize(
        fun = ou_negative_log_likelihood_transformed, 
        x0 = x_not,
        args = (x, dt, var_floor),
        method = "L-BFGS-B"
    )
    
    log_alpha_hat, theta_hat, log_sigma_hat = result.x

    alpha_hat = float(np.exp(log_alpha_hat))
    sigma_hat = float(np.exp(log_sigma_hat))
    theta_hat = float(theta_hat)

    neg_loglik = float(result.fun)
    loglik = neg_loglik * -1
    half_life = compute_half_life(alpha_hat)

    return OUFitResult(
        alpha = alpha_hat,
        theta = theta_hat,
        sigma = sigma_hat,
        log_likelihood = loglik,
        neg_log_likelihood = neg_loglik,
        half_life_days = half_life,
        n_obs = len(x),
        success = bool(result.success),
        message = str(result.message)
    )