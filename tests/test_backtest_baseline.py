from __future__ import annotations

import numpy as np
import pandas as pd

from src.sigstop.backtest.baseline import (
    BaselineBacktestStrategy,
    BaselineStrategyConfig,
    build_baseline_indicator_frame,
    build_baseline_strategy_config,
)
from src.sigstop.backtest.engine import BacktestEngineConfig, StrategyAction, run_backtest_engine


def _toy_window(spread: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods = len(spread), freq = "B"),
            "spread": spread,
        }
    )


def _toy_config() -> dict:
    return {
        "baseline": {
            "ma_window": 3,
            "k": 0.1,
            "min_periods": 3,
            "use_shift_1": True,
        }
    }


def test_build_baseline_strategy_config_reads_backtest_defaults() -> None:
    strategy_config = build_baseline_strategy_config(_toy_config())

    assert strategy_config == BaselineStrategyConfig(
        ma_window = 3,
        k = 0.1,
        min_periods = 3,
        use_shift_1 = True,
        std_ddof = 1,
    )


def test_build_baseline_indicator_frame_uses_strict_shifted_prior_window() -> None:
    trading_window = _toy_window([10.0, 10.0, 10.0, 100.0, 11.0])
    shifted = build_baseline_indicator_frame(
        trading_window,
        strategy_config = BaselineStrategyConfig(
            ma_window = 3,
            k = 0.1,
            min_periods = 3,
            use_shift_1 = True,
        ),
    )
    unshifted = build_baseline_indicator_frame(
        trading_window,
        strategy_config = BaselineStrategyConfig(
            ma_window = 3,
            k = 0.1,
            min_periods = 3,
            use_shift_1 = False,
        ),
    )

    assert shifted["lookback_ready"].tolist() == [False, False, False, True, True]
    assert shifted.loc[3, "ma"] == 10.0
    assert shifted.loc[3, "std"] == 0.0
    assert shifted.loc[3, "upper_band"] == 10.0
    assert bool(shifted.loc[3, "exit_signal"]) is True

    assert unshifted.loc[3, "ma"] == 40.0
    assert np.isclose(unshifted.loc[3, "std"], np.std([10.0, 10.0, 100.0], ddof = 1))
    assert shifted.loc[3, "ma"] != unshifted.loc[3, "ma"]


def test_baseline_strategy_enters_and_exits_on_shifted_bands_with_same_day_execution() -> None:
    strategy = BaselineBacktestStrategy(config = _toy_config())
    result = run_backtest_engine(
        _toy_window([10.0, 10.0, 10.0, 9.0, 10.0, 10.0]),
        strategy,
        engine_config = BacktestEngineConfig(trading_days = 6, execution_price = "close"),
    )

    assert [record.action for record in result.step_records] == [
        StrategyAction.HOLD,
        StrategyAction.HOLD,
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
        StrategyAction.EXIT_LONG_SPREAD,
        StrategyAction.HOLD,
    ]

    entry_metadata = result.step_records[3].decision_metadata
    exit_metadata = result.step_records[4].decision_metadata

    assert entry_metadata["strategy"] == "baseline"
    assert entry_metadata["lookback_ready"] is True
    assert entry_metadata["lookback_start_index"] == 0
    assert entry_metadata["lookback_end_index"] == 2
    assert entry_metadata["ma"] == 10.0
    assert entry_metadata["lower_band"] == 10.0
    assert entry_metadata["execution_price"] == "close"

    assert exit_metadata["lookback_ready"] is True
    assert np.isclose(exit_metadata["ma"], np.mean([10.0, 10.0, 9.0]))
    assert np.isclose(exit_metadata["std"], np.std([10.0, 10.0, 9.0], ddof = 1))
    assert exit_metadata["exit_signal"] is True

    assert strategy.indicator_frame is not None
    assert strategy.indicator_frame["entry_signal"].tolist() == [False, False, False, True, False, False]
