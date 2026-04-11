from __future__ import annotations
from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any
import pandas as pd
from src.sigstop.backtest.costs import BacktestCostConfig, build_backtest_cost_config, compute_action_cost
from src.sigstop.backtest.engine import BacktestEngineResult, BacktestStepRecord, StrategyAction, validate_trading_window


@dataclass(frozen = True)
class TradeLedgerRecord:
    trade_id: int
    strategy: str
    pair: str | None
    beta: float | None
    execution_price: str
    entry_date: str | None
    exit_date: str | None
    entry_idx: int
    exit_idx: int
    entry_spread: float
    exit_spread: float
    gross_pnl_spread: float
    cost_entry: float
    cost_exit: float
    total_cost: float
    net_pnl: float
    holding_days: int
    forced_exit: bool
    forced_exit_reason: str | None
    policy_id_entry: str | None
    policy_id_exit: str | None
    entry_decision_metadata_json: str | None = None
    exit_decision_metadata_json: str | None = None

    # Convert one trade ledger dataclass into dictionary form
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen = True)
class TradeLedgerResult:
    records: list[TradeLedgerRecord]
    trade_ledger: pd.DataFrame


# Serialize decision metadata into a stable JSON string
def _serialize_metadata(metadata: dict[str, Any] | None) -> str | None:
    if metadata is None or not metadata:
        return None
    return json.dumps(metadata, sort_keys = True, ensure_ascii = True)


# Resolve the optional notional reference used by the cost model
def _resolve_notional_reference(metadata: dict[str, Any] | None) -> float | None:
    if metadata is None:
        return None
    candidate = metadata.get("notional_reference")
    if candidate is None:
        return None
    return float(candidate)


# Build the closed-trade ledger from one engine run
def build_trade_ledger(
    engine_result: BacktestEngineResult,
    trading_window: pd.DataFrame,
    *,
    config: dict[str, Any] | None = None,
    cost_config: BacktestCostConfig | None = None,
    pair: str | None = None,
    beta: float | None = None,
) -> TradeLedgerResult:
    resolved_cost_config = (
        cost_config
        if cost_config is not None
        else build_backtest_cost_config(config or {})
    )
    resolved_window = validate_trading_window(trading_window)
    processed_window = resolved_window.iloc[: engine_result.trading_window_length].reset_index(drop = True)

    current_entry_record: BacktestStepRecord | None = None
    latest_position_record: BacktestStepRecord | None = None
    trade_records: list[TradeLedgerRecord] = []

    for step_record in engine_result.step_records:
        if step_record.state_before.value == "LONG_SPREAD":
            latest_position_record = step_record

        if step_record.action == StrategyAction.ENTER_LONG_SPREAD:
            if current_entry_record is not None:
                raise ValueError("Encountered overlapping entry actions while building trade ledger.")
            current_entry_record = step_record
            latest_position_record = None
            continue

        if step_record.action == StrategyAction.EXIT_LONG_SPREAD:
            if current_entry_record is None:
                raise ValueError("Encountered exit action without a matching open entry.")
            trade_records.append(
                _build_trade_record(
                    trade_id = len(trade_records) + 1,
                    strategy = engine_result.strategy_name,
                    execution_price = engine_result.engine_config.execution_price,
                    entry_record = current_entry_record,
                    exit_record = step_record,
                    exit_metadata = step_record.decision_metadata,
                    cost_config = resolved_cost_config,
                    pair = pair,
                    beta = beta,
                    forced_exit = False,
                    forced_exit_reason = None,
                )
            )
            current_entry_record = None
            latest_position_record = None

    if current_entry_record is not None:
        if not engine_result.engine_config.allow_forced_exit_end:
            raise ValueError(
                "Backtest finished with an open position and allow_forced_exit_end=False."
            )

        if processed_window.empty:
            raise ValueError("Cannot force-close an open trade on an empty trading window.")

        final_row = processed_window.iloc[-1]
        forced_exit_record = BacktestStepRecord(
            day_index = int(processed_window.index[-1]),
            date = None if "date" not in processed_window.columns else str(pd.Timestamp(final_row["date"]).date()),
            spread = float(final_row["spread"]),
            hook_name = "forced_exit_end",
            state_before = engine_result.terminal_state,
            action = StrategyAction.EXIT_LONG_SPREAD,
            state_after = engine_result.terminal_state,
            state_start_index_before = engine_result.terminal_state_start_index,
            state_start_index_after = engine_result.terminal_state_start_index,
            position_entry_index = current_entry_record.day_index,
            decision_metadata = {} if latest_position_record is None else dict(latest_position_record.decision_metadata),
        )
        trade_records.append(
            _build_trade_record(
                trade_id = len(trade_records) + 1,
                strategy = engine_result.strategy_name,
                execution_price = engine_result.engine_config.execution_price,
                entry_record = current_entry_record,
                exit_record = forced_exit_record,
                exit_metadata = forced_exit_record.decision_metadata,
                cost_config = resolved_cost_config,
                pair = pair,
                beta = beta,
                forced_exit = True,
                forced_exit_reason = "end_of_horizon",
            )
        )

    return TradeLedgerResult(
        records = trade_records,
        trade_ledger = build_trade_ledger_frame(trade_records),
    )


# Convert trade ledger records into a DataFrame
def build_trade_ledger_frame(records: list[TradeLedgerRecord]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns = [field.name for field in fields(TradeLedgerRecord)])
    return pd.DataFrame([record.to_dict() for record in records])


# Save the trade ledger to CSV or Parquet
def save_trade_ledger(trade_ledger: pd.DataFrame, path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)
    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        trade_ledger.to_csv(resolved_path, index = False)
    elif suffix == ".parquet":
        trade_ledger.to_parquet(resolved_path, index = False)
    else:
        raise ValueError("Trade ledger output path must end with .csv or .parquet.")
    return resolved_path


# Build one closed-trade ledger record from entry and exit events
def _build_trade_record(
    *,
    trade_id: int,
    strategy: str,
    execution_price: str,
    entry_record: BacktestStepRecord,
    exit_record: BacktestStepRecord,
    exit_metadata: dict[str, Any] | None,
    cost_config: BacktestCostConfig,
    pair: str | None,
    beta: float | None,
    forced_exit: bool,
    forced_exit_reason: str | None,
) -> TradeLedgerRecord:
    entry_metadata = dict(entry_record.decision_metadata)
    resolved_exit_metadata = {} if exit_metadata is None else dict(exit_metadata)

    entry_notional_reference = _resolve_notional_reference(entry_metadata)
    exit_notional_reference = _resolve_notional_reference(resolved_exit_metadata)
    cost_entry = compute_action_cost(
        reference_spread = entry_record.spread,
        cost_config = cost_config,
        notional_reference = entry_notional_reference,
    )
    cost_exit = compute_action_cost(
        reference_spread = exit_record.spread,
        cost_config = cost_config,
        notional_reference = exit_notional_reference,
    )

    gross_pnl = float(exit_record.spread - entry_record.spread)
    total_cost = float(cost_entry + cost_exit)
    net_pnl = float(gross_pnl - total_cost)
    holding_days = int(exit_record.day_index - entry_record.day_index)

    if holding_days < 0:
        raise ValueError(
            "Trade exit index must be greater than or equal to entry index. "
            f"Got entry_idx={entry_record.day_index}, exit_idx={exit_record.day_index}"
        )

    return TradeLedgerRecord(
        trade_id = int(trade_id),
        strategy = str(strategy),
        pair = pair,
        beta = None if beta is None else float(beta),
        execution_price = str(execution_price),
        entry_date = entry_record.date,
        exit_date = exit_record.date,
        entry_idx = int(entry_record.day_index),
        exit_idx = int(exit_record.day_index),
        entry_spread = float(entry_record.spread),
        exit_spread = float(exit_record.spread),
        gross_pnl_spread = gross_pnl,
        cost_entry = float(cost_entry),
        cost_exit = float(cost_exit),
        total_cost = total_cost,
        net_pnl = net_pnl,
        holding_days = holding_days,
        forced_exit = bool(forced_exit),
        forced_exit_reason = forced_exit_reason,
        policy_id_entry = _coerce_optional_string(entry_metadata.get("policy_id")),
        policy_id_exit = _coerce_optional_string(resolved_exit_metadata.get("policy_id")),
        entry_decision_metadata_json = _serialize_metadata(entry_metadata),
        exit_decision_metadata_json = _serialize_metadata(resolved_exit_metadata),
    )


# Coerce optional manifest fields into strings
def _coerce_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
