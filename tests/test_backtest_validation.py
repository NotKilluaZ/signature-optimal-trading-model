from __future__ import annotations

import pandas as pd
import pytest

from src.sigstop.backtest.accounting import build_trade_ledger
from src.sigstop.backtest.baseline import BaselineStrategyConfig, build_baseline_indicator_frame
from src.sigstop.backtest.costs import BacktestCostConfig
from src.sigstop.backtest.engine import (
    BacktestDayContext,
    BacktestEngineConfig,
    BacktestEngineResult,
    BacktestState,
    BacktestStepRecord,
    StrategyAction,
    StrategyDecision,
    run_backtest_engine,
)
from src.sigstop.backtest.validation import (
    build_backtest_validation_report,
    validate_baseline_indicator_no_lookahead,
    validate_no_overlapping_positions,
    validate_trade_accounting_identities,
    validate_trade_event_ordering,
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
            return StrategyDecision.enter_long_spread({"policy_id": "entry_policy"})
        return StrategyAction.HOLD

    def on_position_day(self, context):
        if context.current_index == 3:
            return StrategyDecision.exit_long_spread({"policy_id": "exit_policy"})
        return StrategyDecision.hold({"policy_id": "exit_policy"})


class AlwaysHoldStrategy:
    name = "test_strategy"

    def on_flat_day(self, context: BacktestDayContext):
        return StrategyAction.HOLD

    def on_position_day(self, context: BacktestDayContext):
        return StrategyAction.HOLD


class EnterAndNeverExitStrategy:
    name = "test_strategy"

    def on_flat_day(self, context: BacktestDayContext):
        if context.current_index == 1:
            return StrategyDecision.enter_long_spread({"policy_id": "entry_policy"})
        return StrategyAction.HOLD

    def on_position_day(self, context: BacktestDayContext):
        return StrategyDecision.hold({"policy_id": "exit_policy"})


def test_validation_report_passes_for_consistent_round_trip_and_accounting() -> None:
    trading_window = _toy_window([10.0, 9.0, 9.5, 11.0, 10.5])
    engine_result = run_backtest_engine(
        trading_window,
        OneRoundTripStrategy(),
        engine_config = BacktestEngineConfig(trading_days = 5),
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

    validate_no_overlapping_positions(engine_result)
    validate_trade_event_ordering(ledger_result.trade_ledger)
    validate_trade_accounting_identities(ledger_result.trade_ledger)

    report = build_backtest_validation_report(
        strategy = "test_strategy",
        engine_result = engine_result,
        trade_ledger = ledger_result.trade_ledger,
        trading_window = trading_window,
    )

    assert report.passed is True
    assert report.checks["no_overlapping_positions"] is True
    assert report.checks["correct_event_ordering"] is True
    assert report.checks["accounting_identities"] is True


def test_validate_baseline_indicator_no_lookahead_passes_for_shifted_window() -> None:
    trading_window = _toy_window([10.0, 10.0, 10.0, 9.0, 10.0])
    strategy_config = BaselineStrategyConfig(ma_window = 3, k = 0.1, min_periods = 3, use_shift_1 = True)
    indicator_frame = build_baseline_indicator_frame(trading_window, strategy_config = strategy_config)

    validate_baseline_indicator_no_lookahead(
        indicator_frame,
        trading_window,
        strategy_config = strategy_config,
    )


def test_validation_report_passes_for_empty_trade_run() -> None:
    trading_window = _toy_window([10.0, 10.0, 10.0, 10.0])
    engine_result = run_backtest_engine(
        trading_window,
        AlwaysHoldStrategy(),
        engine_config = BacktestEngineConfig(trading_days = 4),
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

    report = build_backtest_validation_report(
        strategy = "test_strategy",
        engine_result = engine_result,
        trade_ledger = ledger_result.trade_ledger,
        trading_window = trading_window,
    )

    assert ledger_result.trade_ledger.empty
    assert report.passed is True
    assert report.details["trade_count"] == 0
    assert report.checks["no_overlapping_positions"] is True
    assert report.checks["correct_event_ordering"] is True
    assert report.checks["accounting_identities"] is True


def test_validation_report_passes_for_forced_end_of_horizon_exit() -> None:
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

    report = build_backtest_validation_report(
        strategy = "test_strategy",
        engine_result = engine_result,
        trade_ledger = ledger_result.trade_ledger,
        trading_window = trading_window,
    )

    assert report.passed is True
    assert bool(ledger_result.trade_ledger.loc[0, "forced_exit"]) is True
    assert ledger_result.trade_ledger.loc[0, "forced_exit_reason"] == "end_of_horizon"
    assert report.checks["correct_event_ordering"] is True


def test_validate_no_overlapping_positions_detects_inconsistent_engine_log() -> None:
    engine_result = BacktestEngineResult(
        strategy_name = "test_strategy",
        engine_config = BacktestEngineConfig(trading_days = 3),
        trading_window_length = 3,
        n_days_processed = 3,
        terminal_state = BacktestState.LONG_SPREAD,
        terminal_state_start_index = 1,
        open_position = None,
        step_records = [
            BacktestStepRecord(
                day_index = 0,
                date = "2022-01-03",
                spread = 10.0,
                hook_name = "on_flat_day",
                state_before = BacktestState.FLAT,
                action = StrategyAction.ENTER_LONG_SPREAD,
                state_after = BacktestState.LONG_SPREAD,
                state_start_index_before = 0,
                state_start_index_after = 0,
                position_entry_index = None,
                decision_metadata = {},
            ),
            BacktestStepRecord(
                day_index = 1,
                date = "2022-01-04",
                spread = 9.0,
                hook_name = "on_position_day",
                state_before = BacktestState.LONG_SPREAD,
                action = StrategyAction.ENTER_LONG_SPREAD,
                state_after = BacktestState.LONG_SPREAD,
                state_start_index_before = 0,
                state_start_index_after = 0,
                position_entry_index = 0,
                decision_metadata = {},
            ),
        ],
    )

    with pytest.raises(ValueError, match = "overlapping positions"):
        validate_no_overlapping_positions(engine_result)


def test_validate_trade_event_ordering_detects_overlapping_trades() -> None:
    overlapping_trade_ledger = pd.DataFrame(
        {
            "trade_id": [1, 2],
            "entry_idx": [1, 3],
            "exit_idx": [4, 5],
            "holding_days": [3, 2],
            "gross_pnl_spread": [1.0, 1.0],
            "cost_entry": [0.0, 0.0],
            "cost_exit": [0.0, 0.0],
            "total_cost": [0.0, 0.0],
            "net_pnl": [1.0, 1.0],
            "forced_exit": [False, False],
        }
    )

    with pytest.raises(ValueError, match = "overlapping or out-of-order trades"):
        validate_trade_event_ordering(overlapping_trade_ledger)


def test_validate_trade_accounting_identities_detects_broken_net_pnl() -> None:
    broken_trade_ledger = pd.DataFrame(
        {
            "trade_id": [1],
            "entry_idx": [1],
            "exit_idx": [3],
            "holding_days": [2],
            "gross_pnl_spread": [2.0],
            "cost_entry": [0.1],
            "cost_exit": [0.2],
            "total_cost": [0.3],
            "net_pnl": [1.8],
        }
    )

    with pytest.raises(ValueError, match = "net_pnl must equal gross_pnl_spread - total_cost"):
        validate_trade_accounting_identities(broken_trade_ledger)
