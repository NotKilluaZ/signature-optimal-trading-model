from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from scripts.build_spread import (
    build_normalized_spread_inputs,
    build_selection_metadata,
    choose_best_orientation_from_map,
)


@dataclass(frozen = True)
class DummyOUFit:
    alpha: float
    theta: float
    sigma: float
    log_likelihood: float
    neg_log_likelihood: float
    half_life_days: float
    n_obs: int
    success: bool
    message: str


@dataclass(frozen = True)
class DummySearchResult:
    orientation: str
    beta_star: float
    log_likelihood_star: float
    ou_fit_star: DummyOUFit
    beta_min: float
    beta_max: float
    grid_points: int
    refine_success: bool
    refine_message: str


def _result(orientation: str, beta_star: float, log_likelihood_star: float) -> DummySearchResult:
    return DummySearchResult(
        orientation = orientation,
        beta_star = beta_star,
        log_likelihood_star = log_likelihood_star,
        ou_fit_star = DummyOUFit(
            alpha = 0.1,
            theta = 1.0,
            sigma = 0.2,
            log_likelihood = log_likelihood_star,
            neg_log_likelihood = -log_likelihood_star,
            half_life_days = 6.5,
            n_obs = 252,
            success = True,
            message = "ok",
        ),
        beta_min = 0.1,
        beta_max = 10.0,
        grid_points = 200,
        refine_success = True,
        refine_message = "ok",
    )


def test_choose_best_orientation_from_map_uses_log_likelihood() -> None:
    gs_result = _result("GS_minus_beta_MS", beta_star = 2.0, log_likelihood_star = -250.0)
    ms_result = _result("MS_minus_beta_GS", beta_star = 0.4, log_likelihood_star = -120.0)

    best_result = choose_best_orientation_from_map(
        {
            gs_result.orientation: gs_result,
            ms_result.orientation: ms_result,
        }
    )

    assert best_result is ms_result


def test_build_selection_metadata_preserves_orientation_specific_values() -> None:
    gs_result = _result("GS_minus_beta_MS", beta_star = 2.0, log_likelihood_star = -250.0)
    ms_result = _result("MS_minus_beta_GS", beta_star = 0.4, log_likelihood_star = -120.0)

    metadata = build_selection_metadata(
        ["GS", "MS"],
        chosen_result = ms_result,
        results_by_orientation = {
            gs_result.orientation: gs_result,
            ms_result.orientation: ms_result,
        },
    )

    assert metadata["pair"] == ["GS", "MS"]
    assert metadata["price_basis"] == "normalized_initial_full_history"
    assert metadata["chosen_orientation"] == "MS_minus_beta_GS"
    assert metadata["chosen_beta"] == 0.4
    assert metadata["orientation_tested"]["GS_minus_beta_MS"]["beta_star"] == 2.0
    assert metadata["orientation_tested"]["GS_minus_beta_MS"]["log_likelihood_star"] == -250.0
    assert metadata["orientation_tested"]["MS_minus_beta_GS"]["beta_star"] == 0.4
    assert metadata["orientation_tested"]["MS_minus_beta_GS"]["log_likelihood_star"] == -120.0


def test_build_normalized_spread_inputs_uses_full_history_scale_for_split_windows() -> None:
    dates = pd.date_range("2021-01-04", periods = 4, freq = "B")
    aligned_df = pd.DataFrame(
        {
            "date": dates,
            "MS": [10.0, 20.0, 40.0, 80.0],
            "GS": [100.0, 200.0, 400.0, 800.0],
        }
    )
    formation_df = aligned_df.iloc[:2].reset_index(drop = True)
    trading_df = aligned_df.iloc[2:].reset_index(drop = True)

    normalized_plot_df, normalized_formation_df, normalized_trading_df, normalized_aligned_df = (
        build_normalized_spread_inputs(
            aligned_df,
            formation_df,
            trading_df,
        )
    )

    assert normalized_plot_df["MS_norm"].tolist() == [1.0, 2.0, 4.0, 8.0]
    assert normalized_plot_df["GS_norm"].tolist() == [1.0, 2.0, 4.0, 8.0]
    assert normalized_aligned_df["MS"].tolist() == [1.0, 2.0, 4.0, 8.0]
    assert normalized_aligned_df["GS"].tolist() == [1.0, 2.0, 4.0, 8.0]
    assert normalized_formation_df["MS"].tolist() == [1.0, 2.0]
    assert normalized_formation_df["GS"].tolist() == [1.0, 2.0]
    assert normalized_trading_df["MS"].tolist() == [4.0, 8.0]
    assert normalized_trading_df["GS"].tolist() == [4.0, 8.0]
