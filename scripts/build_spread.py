from __future__ import annotations
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import pandas as pd
from src.sigstop.config import load_config
from src.sigstop.paths import PROCESSED_DATA_DIR, ensure_directories
from src.sigstop.spread.beta_search import search_best_beta_for_orientation
from src.sigstop.spread.construction import VALID_ORIENTATIONS, construct_normalized_prices, construct_spread
from src.sigstop.spread.diagnostics import build_spread_diagnostics_summary

# Save json to file path
def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as f:
        json.dump(data, f, indent = 2)

# Create matplotlib visual plot for normalized prices of GS and MS stock data
def plot_normalized_prices(normalized_df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize = (10, 5))
    plt.plot(normalized_df["date"], normalized_df["MS_norm"], label = "MS")
    plt.plot(normalized_df["date"], normalized_df["GS_norm"], label = "GS")
    plt.title("MS-GS normalized daily prices")
    plt.xlabel("Date")
    plt.ylabel("Normalized Price")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(output_path)
    plt.close()

# Create spread visual for the spread between MS and GS
def plot_spread(spread_df: pd.DataFrame, title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(spread_df["date"], spread_df["spread"])
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.tight_layout()
    output_path.parent.mkdir(parents = True, exist_ok = True)
    plt.savefig(output_path)
    plt.close()

def run_orientation_searches(
    formation_df: pd.DataFrame,
    *,
    min_factor: float,
    max_factor: float,
    grid_points: int,
    dt_days: float,
    var_floor: float,
    refine_max_iter: int,
    refine_tol: float,
) -> dict[str, Any]:
    results_by_orientation: dict[str, Any] = {}

    for orientation in VALID_ORIENTATIONS:
        results_by_orientation[orientation] = search_best_beta_for_orientation(
            formation_df,
            orientation,
            min_factor,
            max_factor,
            grid_points,
            dt_days,
            var_floor,
            refine_max_iter,
            refine_tol,
        )

    return results_by_orientation


def choose_best_orientation_from_map(results_by_orientation: dict[str, Any]) -> Any:
    if not results_by_orientation:
        raise ValueError("At least one orientation result is required.")
    return max(results_by_orientation.values(), key = lambda result: result.log_likelihood_star)


def build_selection_metadata(
    pair: list[str],
    *,
    chosen_result: Any,
    results_by_orientation: dict[str, Any],
) -> dict[str, Any]:
    orientation_tested: dict[str, Any] = {}
    for orientation in VALID_ORIENTATIONS:
        result = results_by_orientation[orientation]
        orientation_tested[orientation] = {
            "beta_star": float(result.beta_star),
            "log_likelihood_star": float(result.log_likelihood_star),
            "ou_fit_star": asdict(result.ou_fit_star),
            "beta_min": float(result.beta_min),
            "beta_max": float(result.beta_max),
            "grid_points": int(result.grid_points),
            "refine_success": bool(result.refine_success),
            "refine_message": str(result.refine_message),
        }

    return {
        "pair": pair,
        "chosen_orientation": chosen_result.orientation,
        "chosen_beta": float(chosen_result.beta_star),
        "chosen_log_likelihood": float(chosen_result.log_likelihood_star),
        "chosen_ou_fit": asdict(chosen_result.ou_fit_star),
        "orientation_tested": orientation_tested,
    }
    
def main() -> None:
    config  = load_config()
    ensure_directories()

    # Load config varaibles
    beta_cfg = config["beta_optimization"]
    ou_cfg = config["ou_fit"]
    diag_cfg = config["diagnostics"]

    grid_points = int(beta_cfg["grid"]["points"])
    min_factor = float(beta_cfg["bounds"]["min_factor"])
    max_factor = float(beta_cfg["bounds"]["max_factor"])
    refine_max_iter = int(beta_cfg["refine"]["max_iter"])
    refine_tol = float(beta_cfg["refine"]["tol"])

    dt_days = float(ou_cfg["dt_days"])
    var_floor = float(ou_cfg["numeric_stability"]["var_floor"])

    adf_alpha = float(diag_cfg["adf_alpha"])
    kpss_alpha = float(diag_cfg["kpss_alpha"])
    half_life_warn_range = tuple(diag_cfg["half_life_warn_range"])
    acf_nlags = int(diag_cfg["acf_nlags"])


    # Create folders to store plots and data from our pipeline
    artifacts_dir = PROCESSED_DATA_DIR / "artifacts"
    plots_dir = artifacts_dir / "plots"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load paths to read from
    aligned_path = PROCESSED_DATA_DIR / "aligned_prices.parquet"
    formation_path = PROCESSED_DATA_DIR / "formation.parquet"
    trading_path = PROCESSED_DATA_DIR / "trading.parquet"

    # Read parquet and turn into DataFrame
    aligned_df = pd.read_parquet(aligned_path)
    formation_df = pd.read_parquet(formation_path)
    trading_df = pd.read_parquet(trading_path)

    # Create and store normalized aligned DataFrame artifact into folder
    normalzied_df = construct_normalized_prices(aligned_df)
    normalzied_path = artifacts_dir / "prices_normalized_init.parquet"
    normalzied_df.to_parquet(normalzied_path, index = False)

    # Run the beta search for highest likelihood value on both orientations then compare
    results_by_orientation = run_orientation_searches(
        formation_df,
        min_factor = min_factor,
        max_factor = max_factor,
        grid_points = grid_points,
        dt_days = dt_days,
        var_floor = var_floor,
        refine_max_iter = refine_max_iter,
        refine_tol = refine_tol,
    )

    best_result = choose_best_orientation_from_map(results_by_orientation)

    # Build spreads
    spread_full = construct_spread(aligned_df, best_result.beta_star, best_result.orientation)
    spread_formation = construct_spread(formation_df, best_result.beta_star, best_result.orientation)
    spread_trading = construct_spread(trading_df, best_result.beta_star, best_result.orientation)

    # Store artifacts
    spread_full_path = artifacts_dir / "spread_full.parquet"
    spread_formation_path = artifacts_dir / "spread_formation.parquet"
    spread_trading_path = artifacts_dir / "spread_trading.parquet"

    spread_full.to_parquet(spread_full_path, index = False)
    spread_formation.to_parquet(spread_formation_path, index = False)
    spread_trading.to_parquet(spread_trading_path, index = False)

    # Run diagnostics on formation spread
    diagnostics = build_spread_diagnostics_summary(
        spread_formation["spread"],
        best_result.ou_fit_star,
        adf_alpha,
        kpss_alpha,
        half_life_warn_range,
        acf_nlags
        )
    
    # Save diagnostics
    diagnostics_path = artifacts_dir / "spread_diagnostics.json"
    save_json(diagnostics, diagnostics_path)

    # Save selection metadata
    selection_metadata = build_selection_metadata(
        ["GS", "MS"],
        chosen_result = best_result,
        results_by_orientation = results_by_orientation,
    )

    selection_path = artifacts_dir / "spread_selection_data.json"
    save_json(selection_metadata, selection_path)

    # Plot the normalized prices and trading spread as PNG
    plot_normalized_prices(normalzied_df, plots_dir / "gs_ms_normalized_prices.png")
    plot_spread(
        spread_df=spread_trading,
        title=(
            f"GS & MS trading spread | "
            f"{best_result.orientation} | beta={best_result.beta_star:.6f}"
        ),
        output_path=plots_dir / "gs_ms_spread_trading.png",
    )

    # Console Summary
    print(f"Chosen orientation: {best_result.orientation}")
    print(f"Chosen beta: {best_result.beta_star:.6f}")
    print(f"Chosen log-likelihood: {best_result.log_likelihood_star:.6f}")
    print(f"Half-life (days): {best_result.ou_fit_star.half_life_days:.4f}")
    print(f"Artifacts saved to: {artifacts_dir}")

if __name__ == "__main__":
    main()
