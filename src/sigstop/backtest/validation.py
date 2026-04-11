from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.sigstop.backtest.baseline import BaselineStrategyConfig
from src.sigstop.backtest.engine import BacktestEngineResult, StrategyAction, validate_trading_window


@dataclass(frozen = True)
class BacktestValidationResult:
    strategy: str
    checks: dict[str, bool]
    details: dict[str, Any]

    # Report whether all validation checks passed
    @property
    def passed(self) -> bool:
        return bool(all(self.checks.values()))

    # Convert one validation report dataclass into dictionary form
    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "passed": self.passed,
            "checks": dict(self.checks),
            "details": dict(self.details),
        }


# Validate that the engine log never opens overlapping positions
def validate_no_overlapping_positions(engine_result: BacktestEngineResult) -> None:
    has_open_position = False
    for step_record in engine_result.step_records:
        if step_record.action == StrategyAction.ENTER_LONG_SPREAD:
            if has_open_position:
                raise ValueError("Detected overlapping positions: received entry while already long spread.")
            has_open_position = True
        elif step_record.action == StrategyAction.EXIT_LONG_SPREAD:
            if not has_open_position:
                raise ValueError("Detected invalid event ordering: received exit while flat.")
            has_open_position = False


# Validate trade ordering and holding-period consistency in the ledger
def validate_trade_event_ordering(trade_ledger: pd.DataFrame) -> None:
    resolved_trade_ledger = _normalize_trade_ledger(trade_ledger)
    if resolved_trade_ledger.empty:
        return

    ordered = resolved_trade_ledger.sort_values(["entry_idx", "exit_idx", "trade_id"]).reset_index(drop = True)
    previous_exit_idx: int | None = None
    for row in ordered.itertuples(index = False):
        entry_idx = int(row.entry_idx)
        exit_idx = int(row.exit_idx)
        forced_exit = bool(row.forced_exit) if hasattr(row, "forced_exit") else False
        holding_days = int(row.holding_days)

        if exit_idx < entry_idx:
            raise ValueError(
                "Trade exit index must not precede entry index. "
                f"Got entry_idx={entry_idx}, exit_idx={exit_idx}"
            )
        if holding_days != exit_idx - entry_idx:
            raise ValueError(
                "holding_days must equal exit_idx - entry_idx. "
                f"Got holding_days={holding_days}, entry_idx={entry_idx}, exit_idx={exit_idx}"
            )
        if holding_days == 0 and not forced_exit:
            raise ValueError("Zero-day holding periods are only allowed for forced end-of-horizon exits.")
        if previous_exit_idx is not None and entry_idx <= previous_exit_idx:
            raise ValueError(
                "Detected overlapping or out-of-order trades in trade ledger. "
                f"Previous exit_idx={previous_exit_idx}, next entry_idx={entry_idx}"
            )
        previous_exit_idx = exit_idx


# Validate the gross, cost, and net PnL identities for each trade
def validate_trade_accounting_identities(
    trade_ledger: pd.DataFrame,
    *,
    atol: float = 1.0e-12,
) -> None:
    resolved_trade_ledger = _normalize_trade_ledger(trade_ledger)
    if resolved_trade_ledger.empty:
        return

    for row in resolved_trade_ledger.itertuples(index = False):
        cost_entry = float(row.cost_entry)
        cost_exit = float(row.cost_exit)
        total_cost = float(row.total_cost)
        gross_pnl = float(row.gross_pnl_spread)
        net_pnl = float(row.net_pnl)

        if cost_entry < 0.0 or cost_exit < 0.0 or total_cost < 0.0:
            raise ValueError("Trade costs must be non-negative.")
        if not np.isclose(total_cost, cost_entry + cost_exit, atol = atol, rtol = 0.0):
            raise ValueError(
                "total_cost must equal cost_entry + cost_exit. "
                f"Got total_cost={total_cost}, cost_entry={cost_entry}, cost_exit={cost_exit}"
            )
        if not np.isclose(net_pnl, gross_pnl - total_cost, atol = atol, rtol = 0.0):
            raise ValueError(
                "net_pnl must equal gross_pnl_spread - total_cost. "
                f"Got net_pnl={net_pnl}, gross_pnl_spread={gross_pnl}, total_cost={total_cost}"
            )


# Validate that baseline indicators use the intended prior-only lookback window
def validate_baseline_indicator_no_lookahead(
    indicator_frame: pd.DataFrame,
    trading_window: pd.DataFrame,
    *,
    strategy_config: BaselineStrategyConfig,
    atol: float = 1.0e-12,
) -> None:
    resolved_window = validate_trading_window(trading_window)
    if not isinstance(indicator_frame, pd.DataFrame):
        raise TypeError(
            "indicator_frame must be a pandas DataFrame. "
            f"Got: {type(indicator_frame).__name__}"
        )

    required_columns = {"ma", "std", "lookback_ready"}
    missing = required_columns.difference(indicator_frame.columns)
    if missing:
        raise ValueError(
            f"indicator_frame is missing required columns: {sorted(missing)}"
        )

    spread = resolved_window["spread"].to_numpy(dtype = np.float64)
    for row_index in range(len(indicator_frame)):
        row = indicator_frame.iloc[row_index]
        if not bool(row["lookback_ready"]):
            continue

        if strategy_config.use_shift_1:
            start = row_index - int(strategy_config.ma_window)
            stop = row_index
        else:
            start = row_index - int(strategy_config.ma_window) + 1
            stop = row_index + 1

        if start < 0 or stop > len(spread):
            raise ValueError(
                f"Baseline lookback window is out of bounds at row_index={row_index}."
            )

        window = spread[start:stop]
        if len(window) < int(strategy_config.min_periods):
            raise ValueError(
                "Baseline indicator marked lookback_ready before satisfying min_periods. "
                f"row_index={row_index}, window_length={len(window)}"
            )

        expected_ma = float(np.mean(window))
        expected_std = float(np.std(window, ddof = int(strategy_config.std_ddof)))
        observed_ma = float(row["ma"])
        observed_std = float(row["std"])

        if not np.isclose(observed_ma, expected_ma, atol = atol, rtol = 0.0):
            raise ValueError(
                f"Baseline moving average uses look-ahead or wrong window at row_index={row_index}. "
                f"Expected {expected_ma}, observed {observed_ma}"
            )
        if not np.isclose(observed_std, expected_std, atol = atol, rtol = 0.0):
            raise ValueError(
                f"Baseline moving std uses look-ahead or wrong window at row_index={row_index}. "
                f"Expected {expected_std}, observed {observed_std}"
            )


# Build the combined validation report for one backtest strategy run
def build_backtest_validation_report(
    *,
    strategy: str,
    engine_result: BacktestEngineResult,
    trade_ledger: pd.DataFrame,
    trading_window: pd.DataFrame,
    indicator_frame: pd.DataFrame | None = None,
    baseline_strategy_config: BaselineStrategyConfig | None = None,
) -> BacktestValidationResult:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {
        "trade_count": int(len(trade_ledger)),
        "terminal_state": engine_result.terminal_state.value,
        "n_days_processed": int(engine_result.n_days_processed),
    }

    try:
        validate_no_overlapping_positions(engine_result)
        checks["no_overlapping_positions"] = True
    except ValueError as exc:
        checks["no_overlapping_positions"] = False
        details["no_overlapping_positions_error"] = str(exc)

    try:
        validate_trade_event_ordering(trade_ledger)
        checks["correct_event_ordering"] = True
    except ValueError as exc:
        checks["correct_event_ordering"] = False
        details["correct_event_ordering_error"] = str(exc)

    try:
        validate_trade_accounting_identities(trade_ledger)
        checks["accounting_identities"] = True
    except ValueError as exc:
        checks["accounting_identities"] = False
        details["accounting_identities_error"] = str(exc)

    if indicator_frame is not None and baseline_strategy_config is not None:
        try:
            validate_baseline_indicator_no_lookahead(
                indicator_frame,
                trading_window,
                strategy_config = baseline_strategy_config,
            )
            checks["baseline_no_lookahead"] = True
        except ValueError as exc:
            checks["baseline_no_lookahead"] = False
            details["baseline_no_lookahead_error"] = str(exc)

    return BacktestValidationResult(
        strategy = strategy,
        checks = checks,
        details = details,
    )


# Save the validation report to JSON
def save_validation_report(report: BacktestValidationResult, path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)
    with open(resolved_path, "w", encoding = "utf-8") as handle:
        json.dump(report.to_dict(), handle, indent = 2)
    return resolved_path


# Validate and normalize the trade ledger input for validation
def _normalize_trade_ledger(trade_ledger: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(trade_ledger, pd.DataFrame):
        raise TypeError(
            "trade_ledger must be a pandas DataFrame. "
            f"Got: {type(trade_ledger).__name__}"
        )
    required_columns = {
        "trade_id",
        "entry_idx",
        "exit_idx",
        "holding_days",
        "gross_pnl_spread",
        "cost_entry",
        "cost_exit",
        "total_cost",
        "net_pnl",
    }
    missing = required_columns.difference(trade_ledger.columns)
    if missing:
        raise ValueError(f"trade_ledger is missing required columns: {sorted(missing)}")
    return trade_ledger.reset_index(drop = True).copy()
