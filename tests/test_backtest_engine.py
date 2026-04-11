from __future__ import annotations

import pandas as pd
import pytest

from src.sigstop.backtest.engine import (
    BacktestEngineConfig,
    BacktestState,
    StrategyAction,
    StrategyDecision,
    run_backtest_engine,
)


def _toy_window(n_days: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods = n_days, freq = "B"),
            "spread": [10.0 + float(index) for index in range(n_days)],
        }
    )


class ScheduledRoundTripStrategy:
    name = "scheduled_round_trip"

    def __init__(self) -> None:
        self.flat_calls: list[tuple[int, int]] = []
        self.position_calls: list[tuple[int, int, int]] = []

    def on_flat_day(self, context):
        self.flat_calls.append((context.current_index, context.state_start_index))
        if context.current_index == 2:
            return StrategyDecision.enter_long_spread({"signal": "enter"})
        return StrategyDecision.hold()

    def on_position_day(self, context):
        assert context.open_position is not None
        self.position_calls.append(
            (
                context.current_index,
                context.state_start_index,
                context.open_position.entry_day_index,
            )
        )
        if context.current_index == 4:
            return StrategyDecision.exit_long_spread({"signal": "exit"})
        return StrategyDecision.hold()


class InvalidFlatExitStrategy:
    def on_flat_day(self, context):
        return StrategyDecision.exit_long_spread()

    def on_position_day(self, context):
        return StrategyDecision.hold()


class EnterAndHoldStrategy:
    def on_flat_day(self, context):
        if context.current_index == 1:
            return StrategyAction.ENTER_LONG_SPREAD
        return StrategyAction.HOLD

    def on_position_day(self, context):
        return None


def test_backtest_engine_runs_daily_state_machine_and_resets_stage_start_index() -> None:
    strategy = ScheduledRoundTripStrategy()
    result = run_backtest_engine(
        _toy_window(),
        strategy,
        engine_config = BacktestEngineConfig(trading_days = 6),
    )

    assert result.strategy_name == "scheduled_round_trip"
    assert result.n_days_processed == 6
    assert result.terminal_state == BacktestState.FLAT
    assert result.open_position is None

    assert strategy.flat_calls == [(0, 0), (1, 0), (2, 0), (5, 5)]
    assert strategy.position_calls == [(3, 2, 2), (4, 2, 2)]

    assert [record.action for record in result.step_records] == [
        StrategyAction.HOLD,
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
        StrategyAction.HOLD,
        StrategyAction.EXIT_LONG_SPREAD,
        StrategyAction.HOLD,
    ]
    assert result.step_records[2].state_before == BacktestState.FLAT
    assert result.step_records[2].state_after == BacktestState.LONG_SPREAD
    assert result.step_records[4].state_before == BacktestState.LONG_SPREAD
    assert result.step_records[4].state_after == BacktestState.FLAT
    assert result.step_records[4].state_start_index_after == 5


def test_backtest_engine_rejects_invalid_exit_action_while_flat() -> None:
    with pytest.raises(ValueError, match = "Invalid action"):
        run_backtest_engine(
            _toy_window(),
            InvalidFlatExitStrategy(),
            engine_config = BacktestEngineConfig(trading_days = 3),
        )


def test_backtest_engine_caps_processing_to_configured_trading_days_and_keeps_open_position() -> None:
    result = run_backtest_engine(
        _toy_window(n_days = 5),
        EnterAndHoldStrategy(),
        engine_config = BacktestEngineConfig(trading_days = 3),
    )

    assert result.trading_window_length == 3
    assert result.n_days_processed == 3
    assert result.terminal_state == BacktestState.LONG_SPREAD
    assert result.open_position is not None
    assert result.open_position.entry_day_index == 1
    assert result.open_position.entry_date == "2022-01-04"
    assert [record.hook_name for record in result.step_records] == [
        "on_flat_day",
        "on_flat_day",
        "on_position_day",
    ]
