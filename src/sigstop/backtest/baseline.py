from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.sigstop.backtest.engine import (
    BacktestDayContext,
    BacktestStrategy,
    StrategyDecision,
    validate_trading_window,
)
from src.sigstop.config import load_config


@dataclass(frozen = True)
class BaselineStrategyConfig:
    ma_window: int = 100
    k: float = 0.1
    min_periods: int = 100
    use_shift_1: bool = True
    std_ddof: int = 1

    # Validate baseline config values after initialization
    def __post_init__(self) -> None:
        if int(self.ma_window) <= 0:
            raise ValueError(f"ma_window must be positive. Got: {self.ma_window}")
        if int(self.min_periods) <= 0:
            raise ValueError(f"min_periods must be positive. Got: {self.min_periods}")
        if int(self.min_periods) > int(self.ma_window):
            raise ValueError(
                "min_periods must be less than or equal to ma_window. "
                f"Got min_periods={self.min_periods}, ma_window={self.ma_window}"
            )
        if not np.isfinite(float(self.k)):
            raise ValueError(f"k must be finite. Got: {self.k}")
        if float(self.k) < 0.0:
            raise ValueError(f"k must be non-negative. Got: {self.k}")
        if int(self.std_ddof) < 0:
            raise ValueError(f"std_ddof must be non-negative. Got: {self.std_ddof}")


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Build the baseline strategy config from the YAML config dictionary
def build_baseline_strategy_config(config: dict[str, Any]) -> BaselineStrategyConfig:
    return BaselineStrategyConfig(
        ma_window = int(_get_config_value(config, ["baseline", "ma_window"], 100)),
        k = float(_get_config_value(config, ["baseline", "k"], 0.1)),
        min_periods = int(_get_config_value(config, ["baseline", "min_periods"], 100)),
        use_shift_1 = bool(_get_config_value(config, ["baseline", "use_shift_1"], True)),
        std_ddof = int(_get_config_value(config, ["baseline", "std_ddof"], 1)),
    )


# Build the shifted rolling MA and std indicator frame for the baseline rule
def build_baseline_indicator_frame(
    trading_window: pd.DataFrame,
    *,
    strategy_config: BaselineStrategyConfig,
) -> pd.DataFrame:
    resolved_window = validate_trading_window(trading_window)
    spread = resolved_window["spread"].astype(np.float64)

    rolling = spread.rolling(
        window = int(strategy_config.ma_window),
        min_periods = int(strategy_config.min_periods),
    )
    moving_average = rolling.mean()
    moving_std = rolling.std(ddof = int(strategy_config.std_ddof))

    if strategy_config.use_shift_1:
        moving_average = moving_average.shift(1)
        moving_std = moving_std.shift(1)

    lookback_ready = moving_average.notna() & moving_std.notna()
    lower_band = moving_average - float(strategy_config.k) * moving_std
    upper_band = moving_average + float(strategy_config.k) * moving_std
    entry_signal = lookback_ready & (spread < lower_band)
    exit_signal = lookback_ready & (spread > upper_band)

    indicator_frame = resolved_window.copy()
    indicator_frame["ma"] = moving_average.astype(np.float64)
    indicator_frame["std"] = moving_std.astype(np.float64)
    indicator_frame["lower_band"] = lower_band.astype(np.float64)
    indicator_frame["upper_band"] = upper_band.astype(np.float64)
    indicator_frame["lookback_ready"] = lookback_ready.astype(bool)
    indicator_frame["entry_signal"] = entry_signal.astype(bool)
    indicator_frame["exit_signal"] = exit_signal.astype(bool)
    return indicator_frame


class BaselineBacktestStrategy(BacktestStrategy):
    name = "baseline"

    # Initialize the baseline backtest strategy from config
    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
    ) -> None:
        if config is not None and config_path is not None:
            raise ValueError("Provide either config or config_path, not both.")

        self.config = (
            dict(config)
            if config is not None
            else (load_config(config_path) if config_path is not None else load_config())
        )
        self.strategy_config = build_baseline_strategy_config(self.config)
        self.indicator_frame: pd.DataFrame | None = None
        self._prepared_spread: np.ndarray | None = None

    # Evaluate the baseline entry rule while flat
    def on_flat_day(self, context: BacktestDayContext) -> StrategyDecision:
        row = self._indicator_row(context)
        metadata = self._build_metadata(context, row)
        if bool(row["entry_signal"]):
            return StrategyDecision.enter_long_spread(metadata)
        return StrategyDecision.hold(metadata)

    # Evaluate the baseline exit rule while long spread
    def on_position_day(self, context: BacktestDayContext) -> StrategyDecision:
        row = self._indicator_row(context)
        metadata = self._build_metadata(context, row)
        if bool(row["exit_signal"]):
            return StrategyDecision.exit_long_spread(metadata)
        return StrategyDecision.hold(metadata)

    # Fetch the prepared indicator row for the current day
    def _indicator_row(self, context: BacktestDayContext) -> pd.Series:
        self._ensure_indicator_frame(context.trading_window)
        assert self.indicator_frame is not None
        return self.indicator_frame.iloc[int(context.current_index)]

    # Build and cache the baseline indicator frame for one trading window
    def _ensure_indicator_frame(self, trading_window: pd.DataFrame) -> None:
        resolved_window = validate_trading_window(trading_window)
        resolved_spread = resolved_window["spread"].to_numpy(dtype = np.float64)

        if self.indicator_frame is None:
            self.indicator_frame = build_baseline_indicator_frame(
                resolved_window,
                strategy_config = self.strategy_config,
            )
            self._prepared_spread = resolved_spread.copy()
            return

        if self._prepared_spread is None or self._prepared_spread.shape != resolved_spread.shape:
            raise ValueError("Baseline strategy was prepared on a different trading window shape.")
        if not np.array_equal(self._prepared_spread, resolved_spread):
            raise ValueError("Baseline strategy cannot be reused across different trading windows.")

    # Build decision metadata for one baseline trading day
    def _build_metadata(
        self,
        context: BacktestDayContext,
        indicator_row: pd.Series,
    ) -> dict[str, Any]:
        current_index = int(context.current_index)
        lookback_ready = bool(indicator_row["lookback_ready"])
        lookback_start_index = None
        lookback_end_index = None
        if lookback_ready:
            if self.strategy_config.use_shift_1:
                lookback_start_index = current_index - int(self.strategy_config.ma_window)
                lookback_end_index = current_index - 1
            else:
                lookback_start_index = current_index - int(self.strategy_config.ma_window) + 1
                lookback_end_index = current_index

        return {
            "strategy": "baseline",
            "ma_window": int(self.strategy_config.ma_window),
            "baseline_k": float(self.strategy_config.k),
            "min_periods": int(self.strategy_config.min_periods),
            "use_shift_1": bool(self.strategy_config.use_shift_1),
            "std_ddof": int(self.strategy_config.std_ddof),
            "execution_price": str(context.engine_config.execution_price),
            "lookback_ready": lookback_ready,
            "lookback_start_index": lookback_start_index,
            "lookback_end_index": lookback_end_index,
            "ma": None if pd.isna(indicator_row["ma"]) else float(indicator_row["ma"]),
            "std": None if pd.isna(indicator_row["std"]) else float(indicator_row["std"]),
            "lower_band": None if pd.isna(indicator_row["lower_band"]) else float(indicator_row["lower_band"]),
            "upper_band": None if pd.isna(indicator_row["upper_band"]) else float(indicator_row["upper_band"]),
            "entry_signal": bool(indicator_row["entry_signal"]),
            "exit_signal": bool(indicator_row["exit_signal"]),
        }
