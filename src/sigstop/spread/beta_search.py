from __future__ import annotations
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from src.sigstop.spread.construction import construct_spread
from src.sigstop.spread.ou import OUFitResult, fit_ou_mle

@dataclass
class BetaEvaluation:
    beta: float
    log_likelihood: float
    ou_fit: OUFitResult

@dataclass
class BetaSearchResult:
    orientation: str
    beta_star: float
    log_likelihood_star: float
    ou_fit_star: OUFitResult
    beta_min: float
    beta_max: float
    grid_points: int
    grid_results: list[BetaEvaluation]
    refine_success: bool
    refine_message: str

# Compute beta bounds to search for possible beta values using formation-window average price ratio
def compute_ratio_based_beta_bounds(formation_df: pd.DataFrame, orientation: str, min_factor: float = 0.1, max_factor: float = 10.0) -> tuple[float, float]:
    if orientation == "GS_minus_beta_MS":
        ratio_not = formation_df["GS"].mean() / formation_df["MS"].mean()
    elif orientation == "MS_minus_beta_GS":
        ratio_not = formation_df["MS"].mean() / formation_df["GS"].mean()
    else:
        raise ValueError(f"Invalid orientation. Got: {orientation}")
    
    beta_min = float(min_factor * ratio_not)
    beta_max = float(max_factor * ratio_not)

    if beta_min <= 0 or beta_max <= 0:
        raise ValueError("Computed beta bounds less than or equal to zero")
    if beta_min >= beta_max:
        raise ValueError("Computed min beta is greater than or equal to beta max")
    
    return beta_min, beta_max

# Construct formation spread for one beta, fit OU model by MLE then return evaluation
def evaluate_beta_for_orientation(formation_df: pd.DataFrame, beta: float, orientation: str, dt: float = 1.0, var_floor: float = 1e-12) -> BetaEvaluation:
    
    spread_df = construct_spread(formation_df, beta, orientation)

    ou_fit = fit_ou_mle(spread_df["spread"], dt, var_floor)

    return BetaEvaluation(
        beta = float(beta),
        log_likelihood = float(ou_fit.log_likelihood),
        ou_fit = ou_fit
    )

# Evaluate beta grid for given orientation
def run_beta_grid_search(formation_df: pd.DataFrame, orientation: str, beta_min: float, beta_max: float, grid_points: int = 200, dt: float = 1.0, var_floor: float = 1e-12) -> list[BetaEvaluation]:
    if grid_points < 2:
        raise ValueError(f"Grid points must be greater than or equal to 2. Got {grid_points}")
    
    beta_grid = np.linspace(beta_min, beta_max, grid_points)
    results: list[BetaEvaluation] = []

    for beta in beta_grid:
        evaluation = evaluate_beta_for_orientation(formation_df, beta, orientation, dt, var_floor)
        results.append(evaluation)

    return results

# Return beta with the greatest log likelihood value
def get_best_grid_result(grid_results: list[BetaEvaluation]) -> BetaEvaluation:
    if not grid_results:
        raise ValueError("Grid results empty")
    
    return max(grid_results, key = lambda x: x.log_likelihood)

# Refine the best beta further using a bounded 1D optimizer by minimizing negative log likelihood
def refine_beta_locally(formation_df: pd.DataFrame, orientation: str, bracket_left: float, bracket_right: float, dt: float = 1.0, var_floor: float = 1e-12, max_iter = 200, tol: float = 1e-6):
    if bracket_left <= 0 or bracket_right <= 0:
        raise ValueError("Refinement bracket must be positive")
    if bracket_left >= bracket_right:
        raise ValueError("Left bracket must be less than right bracket")
    
    def objective(beta: float) -> float:
        evaluation = evaluate_beta_for_orientation(formation_df, float(beta), orientation, dt, var_floor)
        return evaluation.log_likelihood * -1
    
    result = minimize_scalar(
        objective, 
        bounds = (bracket_left, bracket_right), 
        method = "bounded", 
        options = {"maxiter": max_iter, "xatol": tol}
    )

    beta_star = float(result.x)
    evaluation_star = evaluate_beta_for_orientation(formation_df, beta_star, orientation, dt, var_floor)

    return result, evaluation_star

# Full beta search pipeline: Compute beta bounds, run grid search, then refine around best beta in grid
def search_best_beta_for_orientation(formation_df: pd.DataFrame, orientation: str, min_factor: float = 0.1, max_factor: float = 10.0, grid_points: int = 200, dt: float = 1.0, var_floor: float = 1e-12, refine_max_iter: int = 200, refine_tol: float = 1e-6) -> BetaSearchResult:
    beta_min, beta_max = compute_ratio_based_beta_bounds(formation_df, orientation, min_factor, max_factor)
    
    grid_results = run_beta_grid_search(formation_df, orientation, beta_min, beta_max, grid_points, dt, var_floor)

    best_grid = get_best_grid_result(grid_results)

    beta_grid = np.linspace(beta_min, beta_max, grid_points)
    best_idx = int(np.argmin(np.abs(beta_grid - best_grid.beta)))

    l = max(0, best_idx - 1)
    r = min(len(beta_grid) - 1, best_idx + 1)

    bracket_left = float(beta_grid[l])
    bracket_right = float(beta_grid[r])

    if bracket_left == bracket_right:
        bracket_left = beta_min
        bracket_right = beta_max

    refine_result, refined_eval = refine_beta_locally(
        formation_df,
        orientation,
        bracket_left,
        bracket_right, 
        dt, 
        var_floor,
        refine_max_iter,
        refine_tol
    )

    return BetaSearchResult(
        orientation = orientation,
        beta_star = refined_eval.beta,
        log_likelihood_star = refined_eval.log_likelihood,
        ou_fit_star = refined_eval.ou_fit,
        beta_min = beta_min,
        beta_max = beta_max,
        grid_points = grid_points,
        grid_results = grid_results,
        refine_success = bool(refine_result.success),
        refine_message = str(refine_result.message)
    )