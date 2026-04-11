from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.sigstop.backtest.metrics import (
    BacktestMetricsConfig,
    build_backtest_metrics_config,
    build_comparison_metrics_table,
    build_equity_curve,
    build_performance_summary,
    compute_annualized_sharpe,
)


def _toy_window(spread: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods = len(spread), freq = "B"),
            "spread": spread,
        }
    )


def _toy_trade_ledger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_id": [1, 2],
            "strategy": ["sot", "sot"],
            "pair": ["GS-MS", "GS-MS"],
            "exit_idx": [1, 3],
            "net_pnl": [0.5, -0.2],
            "holding_days": [2, 1],
            "forced_exit": [False, False],
        }
    )


def test_build_backtest_metrics_config_reads_defaults() -> None:
    config = build_backtest_metrics_config(
        {
            "backtest": {"annualization_factor": 252},
            "metrics": {"rf_rate_daily": 0.001},
        }
    )

    assert config == BacktestMetricsConfig(
        rf_rate_daily = 0.001,
        annualization_factor = 252,
        initial_equity = 1.0,
    )


def test_build_equity_curve_accumulates_daily_realized_trade_pnl() -> None:
    equity_curve = build_equity_curve(
        _toy_trade_ledger(),
        _toy_window([10.0, 10.5, 10.25, 10.75, 10.6]),
        strategy = "sot",
        initial_equity = 1.0,
    )

    np.testing.assert_allclose(equity_curve["daily_net_pnl"], [0.0, 0.5, 0.0, -0.2, 0.0])
    np.testing.assert_allclose(equity_curve["cumulative_net_pnl"], [0.0, 0.5, 0.5, 0.3, 0.3])
    np.testing.assert_allclose(equity_curve["equity"], [1.0, 1.5, 1.5, 1.3, 1.3])
    np.testing.assert_allclose(equity_curve["daily_return"], [0.0, 0.5, 0.0, -0.1333333333333333, 0.0])
    np.testing.assert_allclose(equity_curve["running_peak"], [1.0, 1.5, 1.5, 1.5, 1.5])
    np.testing.assert_allclose(equity_curve["drawdown"], [0.0, 0.0, 0.0, -0.1333333333333333, -0.1333333333333333])
    np.testing.assert_array_equal(equity_curve["cumulative_trade_count"], [0, 1, 1, 2, 2])


def test_performance_summary_computes_trade_and_curve_metrics() -> None:
    trade_ledger = _toy_trade_ledger()
    equity_curve = build_equity_curve(
        trade_ledger,
        _toy_window([10.0, 10.5, 10.25, 10.75, 10.6]),
        strategy = "sot",
        initial_equity = 1.0,
    )
    metrics_config = BacktestMetricsConfig(rf_rate_daily = 0.0, annualization_factor = 252, initial_equity = 1.0)
    summary = build_performance_summary(
        trade_ledger,
        equity_curve,
        metrics_config = metrics_config,
    )

    expected_sharpe = compute_annualized_sharpe(
        equity_curve["daily_return"],
        rf_rate_daily = 0.0,
        annualization_factor = 252,
    )

    assert summary.strategy == "sot"
    assert summary.pair == "GS-MS"
    assert summary.trade_count == 2
    assert summary.win_rate == pytest.approx(0.5)
    assert summary.average_holding_days == pytest.approx(1.5)
    assert summary.max_drawdown == pytest.approx(-0.1333333333333333)
    assert summary.sharpe_ann == pytest.approx(expected_sharpe)
    assert summary.total_net_pnl == pytest.approx(0.3)
    assert summary.final_cumulative_net_pnl == pytest.approx(0.3)
    assert summary.final_equity == pytest.approx(1.3)
    assert summary.total_return == pytest.approx(0.3)
    assert summary.annualized_return is not None


def test_performance_summary_handles_no_trade_case_with_flat_equity_curve() -> None:
    empty_trade_ledger = pd.DataFrame(
        columns = ["exit_idx", "net_pnl", "holding_days", "strategy", "pair", "forced_exit"]
    )
    equity_curve = build_equity_curve(
        empty_trade_ledger,
        _toy_window([10.0, 10.0, 10.0]),
        strategy = "baseline",
        initial_equity = 1.0,
    )
    summary = build_performance_summary(
        empty_trade_ledger,
        equity_curve,
        metrics_config = BacktestMetricsConfig(),
        strategy = "baseline",
        pair = "GS-MS",
    )

    assert summary.trade_count == 0
    assert summary.win_rate is None
    assert summary.average_holding_days is None
    assert summary.sharpe_ann is None
    assert summary.max_drawdown == pytest.approx(0.0)
    assert summary.final_equity == pytest.approx(1.0)
    assert summary.total_net_pnl == pytest.approx(0.0)


def test_build_comparison_metrics_table_stacks_strategy_summaries() -> None:
    metrics_table = build_comparison_metrics_table(
        [
            build_performance_summary(
                _toy_trade_ledger(),
                build_equity_curve(
                    _toy_trade_ledger(),
                    _toy_window([10.0, 10.5, 10.25, 10.75, 10.6]),
                    strategy = "sot",
                    initial_equity = 1.0,
                ),
                metrics_config = BacktestMetricsConfig(),
            ),
            build_performance_summary(
                pd.DataFrame(
                    {
                        "trade_id": [1],
                        "strategy": ["baseline"],
                        "pair": ["GS-MS"],
                        "exit_idx": [2],
                        "net_pnl": [0.1],
                        "holding_days": [1],
                        "forced_exit": [False],
                    }
                ),
                build_equity_curve(
                    pd.DataFrame(
                        {
                            "trade_id": [1],
                            "strategy": ["baseline"],
                            "pair": ["GS-MS"],
                            "exit_idx": [2],
                            "net_pnl": [0.1],
                            "holding_days": [1],
                            "forced_exit": [False],
                        }
                    ),
                    _toy_window([10.0, 9.9, 10.1]),
                    strategy = "baseline",
                    initial_equity = 1.0,
                ),
                metrics_config = BacktestMetricsConfig(),
            ),
        ]
    )

    assert metrics_table["strategy"].tolist() == ["sot", "baseline"]
    assert "sharpe_ann" in metrics_table.columns
    assert "max_drawdown" in metrics_table.columns
    assert metrics_table.loc[0, "trade_count"] == 2
    assert metrics_table.loc[1, "trade_count"] == 1
