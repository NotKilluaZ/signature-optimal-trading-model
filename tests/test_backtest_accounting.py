from __future__ import annotations

import pandas as pd
import pytest

from src.sigstop.backtest.accounting import PairTradeAccountingSpec, build_trade_ledger, save_trade_ledger
from src.sigstop.backtest.costs import BacktestCostConfig, build_backtest_cost_config, compute_action_cost
from src.sigstop.backtest.engine import (
    BacktestEngineConfig,
    StrategyAction,
    StrategyDecision,
    run_backtest_engine,
)


def _toy_window(spread: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods = len(spread), freq = "B"),
            "spread": spread,
        }
    )


class OneRoundTripStrategy:
    name = "test_strategy"

    def on_flat_day(self, context):
        if context.current_index == 1:
            return StrategyDecision.enter_long_spread({"policy_id": "entry_policy_a", "signal": "entry"})
        return StrategyAction.HOLD

    def on_position_day(self, context):
        if context.current_index == 3:
            return StrategyDecision.exit_long_spread({"policy_id": "exit_policy_b", "signal": "exit"})
        return StrategyDecision.hold({"policy_id": "exit_policy_b", "signal": "hold_exit"})


class EnterAndNeverExitStrategy:
    name = "test_strategy"

    def on_flat_day(self, context):
        if context.current_index == 1:
            return StrategyDecision.enter_long_spread({"policy_id": "entry_policy_open"})
        return StrategyAction.HOLD

    def on_position_day(self, context):
        return StrategyDecision.hold({"policy_id": "exit_policy_open"})


def test_cost_config_and_action_cost_follow_spread_proxy_defaults() -> None:
    cost_config = build_backtest_cost_config(
        {
            "costs": {
                "fixed_per_action": 0.1,
                "proportional": 0.01,
                "slippage_bps": 10.0,
                "model": "spread_proxy",
            }
        }
    )

    assert cost_config == BacktestCostConfig(
        fixed_per_action = 0.1,
        proportional = 0.01,
        slippage_bps = 10.0,
        model = "spread_proxy",
    )
    assert compute_action_cost(reference_spread = 20.0, cost_config = cost_config) == pytest.approx(0.32)


def test_trade_ledger_builds_closed_trade_with_costs_and_policy_ids() -> None:
    engine_result = run_backtest_engine(
        _toy_window([10.0, 9.0, 10.0, 12.0, 11.0]),
        OneRoundTripStrategy(),
        engine_config = BacktestEngineConfig(trading_days = 5, execution_price = "close"),
    )
    ledger_result = build_trade_ledger(
        engine_result,
        _toy_window([10.0, 9.0, 10.0, 12.0, 11.0]),
        cost_config = BacktestCostConfig(
            fixed_per_action = 0.1,
            proportional = 0.01,
            slippage_bps = 0.0,
            model = "spread_proxy",
        ),
        pair = "GS-MS",
        beta = 1.25,
    )

    assert len(ledger_result.records) == 1
    record = ledger_result.records[0]
    assert record.trade_id == 1
    assert record.strategy == "test_strategy"
    assert record.pair == "GS-MS"
    assert record.beta == pytest.approx(1.25)
    assert record.entry_date == "2022-01-04"
    assert record.exit_date == "2022-01-06"
    assert record.entry_idx == 1
    assert record.exit_idx == 3
    assert record.entry_spread == pytest.approx(9.0)
    assert record.exit_spread == pytest.approx(12.0)
    assert record.gross_pnl_spread == pytest.approx(3.0)
    assert record.cost_entry == pytest.approx(0.19)
    assert record.cost_exit == pytest.approx(0.22)
    assert record.total_cost == pytest.approx(0.41)
    assert record.net_pnl == pytest.approx(2.59)
    assert record.holding_days == 2
    assert record.forced_exit is False
    assert record.policy_id_entry == "entry_policy_a"
    assert record.policy_id_exit == "exit_policy_b"

    frame = ledger_result.trade_ledger
    assert list(frame.columns[:8]) == [
        "trade_id",
        "strategy",
        "pair",
        "beta",
        "execution_price",
        "entry_date",
        "exit_date",
        "entry_idx",
    ]
    assert frame.loc[0, "net_pnl"] == pytest.approx(2.59)


def test_trade_ledger_force_closes_terminal_open_position_and_reuses_latest_exit_policy_context() -> None:
    trading_window = _toy_window([10.0, 9.0, 8.5, 8.0])
    engine_result = run_backtest_engine(
        trading_window,
        EnterAndNeverExitStrategy(),
        engine_config = BacktestEngineConfig(trading_days = 4, allow_forced_exit_end = True),
    )
    ledger_result = build_trade_ledger(
        engine_result,
        trading_window,
        cost_config = BacktestCostConfig(
            fixed_per_action = 0.0,
            proportional = 0.0,
            slippage_bps = 0.0,
            model = "spread_proxy",
        ),
    )

    assert len(ledger_result.records) == 1
    record = ledger_result.records[0]
    assert record.forced_exit is True
    assert record.forced_exit_reason == "end_of_horizon"
    assert record.entry_idx == 1
    assert record.exit_idx == 3
    assert record.holding_days == 2
    assert record.policy_id_entry == "entry_policy_open"
    assert record.policy_id_exit == "exit_policy_open"
    assert record.net_pnl == pytest.approx(-1.0)

    output_path = save_trade_ledger(
        ledger_result.trade_ledger,
        "runs/pytest_artifacts/backtest_accounting_forced_exit/trades.csv",
    )
    assert output_path.exists()


def test_trade_ledger_supports_paper_capital_pair_accounting() -> None:
    trading_window = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods = 5, freq = "B"),
            "spread": [50.0, 50.0, 51.0, 54.0, 54.0],
            "GS": [100.0, 100.0, 102.0, 105.0, 105.0],
            "MS": [50.0, 50.0, 51.0, 51.0, 51.0],
        }
    )
    engine_result = run_backtest_engine(
        trading_window[["date", "spread"]],
        OneRoundTripStrategy(),
        engine_config = BacktestEngineConfig(trading_days = 5, execution_price = "close"),
    )
    ledger_result = build_trade_ledger(
        engine_result,
        trading_window,
        cost_config = BacktestCostConfig(
            fixed_per_action = 0.0,
            proportional = 0.0,
            slippage_bps = 0.0,
            model = "spread_proxy",
        ),
        pair_trade_accounting = PairTradeAccountingSpec(
            leg_1_symbol = "GS",
            leg_2_symbol = "MS",
            beta = 1.0,
        ),
    )

    assert len(ledger_result.records) == 1
    record = ledger_result.records[0]
    assert record.accounting_model == "paper_capital_pair"
    assert record.position_units == pytest.approx(0.01)
    assert record.entry_equity == pytest.approx(1.0)
    assert record.exit_equity == pytest.approx(1.04)
    assert record.gross_pnl_spread == pytest.approx(4.0)
    assert record.gross_pnl_capital == pytest.approx(0.04)
    assert record.net_pnl == pytest.approx(0.04)


def test_trade_ledger_raises_when_open_position_remains_and_forced_exit_is_disabled() -> None:
    trading_window = _toy_window([10.0, 9.0, 8.5, 8.0])
    engine_result = run_backtest_engine(
        trading_window,
        EnterAndNeverExitStrategy(),
        engine_config = BacktestEngineConfig(trading_days = 4, allow_forced_exit_end = False),
    )

    with pytest.raises(ValueError, match = "allow_forced_exit_end=False"):
        build_trade_ledger(
            engine_result,
            trading_window,
            cost_config = BacktestCostConfig(
                fixed_per_action = 0.0,
                proportional = 0.0,
                slippage_bps = 0.0,
                model = "spread_proxy",
            ),
        )
