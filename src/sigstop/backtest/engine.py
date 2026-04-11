from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable
import numpy as np
import pandas as pd


class BacktestState(str, Enum):
    FLAT = "FLAT"
    LONG_SPREAD = "LONG_SPREAD"


class StrategyAction(str, Enum):
    HOLD = "hold"
    ENTER_LONG_SPREAD = "enter_long_spread"
    EXIT_LONG_SPREAD = "exit_long_spread"


@dataclass(frozen = True)
class BacktestEngineConfig:
    trading_days: int = 252
    execution_price: str = "close"
    allow_forced_exit_end: bool = True
    annualization_factor: int = 252

    # Validate engine config values after initialization
    def __post_init__(self) -> None:
        if int(self.trading_days) <= 0:
            raise ValueError(f"trading_days must be positive. Got: {self.trading_days}")
        if str(self.execution_price) not in {"close", "next_open"}:
            raise ValueError(
                "execution_price must be either 'close' or 'next_open'. "
                f"Got: {self.execution_price!r}"
            )
        if int(self.annualization_factor) <= 0:
            raise ValueError(
                f"annualization_factor must be positive. Got: {self.annualization_factor}"
            )


@dataclass(frozen = True)
class TradingDayObservation:
    day_index: int
    date: str | None
    spread: float
    row_data: dict[str, Any] = field(default_factory = dict)


@dataclass(frozen = True)
class OpenPosition:
    entry_day_index: int
    entry_date: str | None
    entry_spread: float
    entry_metadata: dict[str, Any] = field(default_factory = dict)


@dataclass(frozen = True)
class BacktestDayContext:
    engine_config: BacktestEngineConfig
    trading_window: pd.DataFrame
    current_day: TradingDayObservation
    state: BacktestState
    state_start_index: int
    completed_round_trips: int
    open_position: OpenPosition | None = None

    # Get the current day index inside the trading window
    @property
    def current_index(self) -> int:
        return int(self.current_day.day_index)

    # Get the processed trading-window length
    @property
    def trading_window_length(self) -> int:
        return int(len(self.trading_window))

    # Get the number of elapsed days in the current state
    @property
    def days_in_state(self) -> int:
        return int(self.current_index - self.state_start_index)

    # Get the number of remaining days including today
    @property
    def remaining_days_inclusive(self) -> int:
        return int(self.trading_window_length - self.current_index)

    # Get the number of remaining days after today
    @property
    def remaining_days_after_today(self) -> int:
        return int(self.trading_window_length - self.current_index - 1)


@dataclass(frozen = True)
class StrategyDecision:
    action: StrategyAction = StrategyAction.HOLD
    metadata: dict[str, Any] = field(default_factory = dict)

    # Build a hold decision with optional metadata
    @classmethod
    def hold(cls, metadata: dict[str, Any] | None = None) -> StrategyDecision:
        return cls(
            action = StrategyAction.HOLD,
            metadata = {} if metadata is None else dict(metadata),
        )

    # Build an enter-long-spread decision with optional metadata
    @classmethod
    def enter_long_spread(
        cls,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyDecision:
        return cls(
            action = StrategyAction.ENTER_LONG_SPREAD,
            metadata = {} if metadata is None else dict(metadata),
        )

    # Build an exit-long-spread decision with optional metadata
    @classmethod
    def exit_long_spread(
        cls,
        metadata: dict[str, Any] | None = None,
    ) -> StrategyDecision:
        return cls(
            action = StrategyAction.EXIT_LONG_SPREAD,
            metadata = {} if metadata is None else dict(metadata),
        )


@dataclass(frozen = True)
class BacktestStepRecord:
    day_index: int
    date: str | None
    spread: float
    hook_name: str
    state_before: BacktestState
    action: StrategyAction
    state_after: BacktestState
    state_start_index_before: int
    state_start_index_after: int
    position_entry_index: int | None
    decision_metadata: dict[str, Any] = field(default_factory = dict)


@dataclass(frozen = True)
class BacktestEngineResult:
    strategy_name: str
    engine_config: BacktestEngineConfig
    trading_window_length: int
    n_days_processed: int
    terminal_state: BacktestState
    terminal_state_start_index: int
    open_position: OpenPosition | None
    step_records: list[BacktestStepRecord]


@runtime_checkable
class BacktestStrategy(Protocol):
    # Handle one flat-state day decision
    def on_flat_day(self, context: BacktestDayContext) -> StrategyDecision:
        ...

    # Handle one in-position day decision
    def on_position_day(self, context: BacktestDayContext) -> StrategyDecision:
        ...


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Build the backtest engine config from the YAML config dictionary
def build_backtest_engine_config(config: dict[str, Any]) -> BacktestEngineConfig:
    trading_days = int(
        _get_config_value(
            config,
            ["backtest", "trading_days"],
            _get_config_value(config, ["split", "trading_days"], 252),
        )
    )
    execution_price = str(
        _get_config_value(config, ["backtest", "execution_price"], "close")
    )
    allow_forced_exit_end = bool(
        _get_config_value(config, ["backtest", "allow_forced_exit_end"], True)
    )
    annualization_factor = int(
        _get_config_value(
            config,
            ["backtest", "annualization_factor"],
            trading_days,
        )
    )
    return BacktestEngineConfig(
        trading_days = trading_days,
        execution_price = execution_price,
        allow_forced_exit_end = allow_forced_exit_end,
        annualization_factor = annualization_factor,
    )


# Validate the trading-window DataFrame and coerce it into clean daily form
def validate_trading_window(trading_window: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(trading_window, pd.DataFrame):
        raise TypeError(
            "trading_window must be a pandas DataFrame. "
            f"Got: {type(trading_window).__name__}"
        )
    if "spread" not in trading_window.columns:
        raise ValueError("trading_window must contain a 'spread' column.")

    resolved = trading_window.reset_index(drop = True).copy()
    if resolved.empty:
        return resolved

    spread = resolved["spread"].to_numpy(dtype = np.float64)
    if not np.all(np.isfinite(spread)):
        raise ValueError("trading_window spread values must be finite.")
    return resolved


# Build one serializable day observation from the trading window
def build_trading_day_observation(
    trading_window: pd.DataFrame,
    day_index: int,
) -> TradingDayObservation:
    row = trading_window.iloc[int(day_index)]
    return TradingDayObservation(
        day_index = int(day_index),
        date = _serialize_date(row["date"]) if "date" in trading_window.columns else None,
        spread = float(row["spread"]),
        row_data = row.to_dict(),
    )


# Serialize a date-like value into ISO day format
def _serialize_date(value: Any) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return str(timestamp.date())


# Normalize strategy hook outputs into one StrategyDecision object
def _coerce_strategy_decision(
    decision: StrategyDecision | StrategyAction | None,
) -> StrategyDecision:
    if decision is None:
        return StrategyDecision.hold()
    if isinstance(decision, StrategyDecision):
        return StrategyDecision(
            action = StrategyAction(decision.action),
            metadata = dict(decision.metadata),
        )
    if isinstance(decision, StrategyAction):
        return StrategyDecision(action = decision)
    raise TypeError(
        "Strategy hooks must return StrategyDecision, StrategyAction, or None. "
        f"Got: {type(decision).__name__}"
    )


# Enforce valid state transitions for the current engine state
def _validate_state_transition(
    state: BacktestState,
    action: StrategyAction,
) -> None:
    allowed_actions = (
        {StrategyAction.HOLD, StrategyAction.ENTER_LONG_SPREAD}
        if state == BacktestState.FLAT
        else {StrategyAction.HOLD, StrategyAction.EXIT_LONG_SPREAD}
    )
    if action not in allowed_actions:
        raise ValueError(
            f"Invalid action {action.value!r} for state {state.value!r}. "
            f"Allowed actions: {[candidate.value for candidate in sorted(allowed_actions, key = str)]}"
        )


# Run the daily FLAT/LONG_SPREAD backtest state machine for one strategy
def run_backtest_engine(
    trading_window: pd.DataFrame,
    strategy: BacktestStrategy,
    *,
    config: dict[str, Any] | None = None,
    engine_config: BacktestEngineConfig | None = None,
) -> BacktestEngineResult:
    if config is None and engine_config is None:
        raise ValueError("Provide either config or engine_config.")

    resolved_engine_config = (
        engine_config if engine_config is not None else build_backtest_engine_config(config or {})
    )
    resolved_window = validate_trading_window(trading_window)
    processed_window = resolved_window.iloc[: resolved_engine_config.trading_days].reset_index(drop = True)

    state = BacktestState.FLAT
    state_start_index = 0
    open_position: OpenPosition | None = None
    completed_round_trips = 0
    step_records: list[BacktestStepRecord] = []

    for day_index in range(len(processed_window)):
        observation = build_trading_day_observation(processed_window, day_index)
        context = BacktestDayContext(
            engine_config = resolved_engine_config,
            trading_window = processed_window,
            current_day = observation,
            state = state,
            state_start_index = state_start_index,
            completed_round_trips = completed_round_trips,
            open_position = open_position,
        )

        if state == BacktestState.FLAT:
            hook_name = "on_flat_day"
            decision = _coerce_strategy_decision(strategy.on_flat_day(context))
        else:
            hook_name = "on_position_day"
            decision = _coerce_strategy_decision(strategy.on_position_day(context))

        _validate_state_transition(state, decision.action)

        state_before = state
        state_start_index_before = state_start_index
        position_entry_index = None if open_position is None else open_position.entry_day_index

        if state == BacktestState.FLAT and decision.action == StrategyAction.ENTER_LONG_SPREAD:
            open_position = OpenPosition(
                entry_day_index = observation.day_index,
                entry_date = observation.date,
                entry_spread = observation.spread,
                entry_metadata = dict(decision.metadata),
            )
            state = BacktestState.LONG_SPREAD
            state_start_index = observation.day_index
        elif state == BacktestState.LONG_SPREAD and decision.action == StrategyAction.EXIT_LONG_SPREAD:
            open_position = None
            state = BacktestState.FLAT
            state_start_index = observation.day_index + 1
            completed_round_trips += 1

        step_records.append(
            BacktestStepRecord(
                day_index = observation.day_index,
                date = observation.date,
                spread = observation.spread,
                hook_name = hook_name,
                state_before = state_before,
                action = decision.action,
                state_after = state,
                state_start_index_before = state_start_index_before,
                state_start_index_after = state_start_index,
                position_entry_index = position_entry_index,
                decision_metadata = dict(decision.metadata),
            )
        )

    return BacktestEngineResult(
        strategy_name = str(getattr(strategy, "name", strategy.__class__.__name__)),
        engine_config = resolved_engine_config,
        trading_window_length = int(len(processed_window)),
        n_days_processed = int(len(processed_window)),
        terminal_state = state,
        terminal_state_start_index = int(state_start_index),
        open_position = open_position,
        step_records = step_records,
    )
