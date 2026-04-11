from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.sigstop.backtest.engine import validate_trading_window


@dataclass(frozen = True)
class BacktestMetricsConfig:
    rf_rate_daily: float = 0.0
    annualization_factor: int = 252
    initial_equity: float = 1.0

    # Validate metrics config values after initialization
    def __post_init__(self) -> None:
        if not np.isfinite(float(self.rf_rate_daily)):
            raise ValueError(f"rf_rate_daily must be finite. Got: {self.rf_rate_daily}")
        if int(self.annualization_factor) <= 0:
            raise ValueError(
                f"annualization_factor must be positive. Got: {self.annualization_factor}"
            )
        if not np.isfinite(float(self.initial_equity)):
            raise ValueError(f"initial_equity must be finite. Got: {self.initial_equity}")


@dataclass(frozen = True)
class BacktestPerformanceSummary:
    strategy: str | None
    pair: str | None
    trade_count: int
    win_rate: float | None
    average_holding_days: float | None
    sharpe_ann: float | None
    max_drawdown: float
    daily_return_mean: float | None
    daily_return_std: float | None
    total_net_pnl: float
    final_cumulative_net_pnl: float
    final_equity: float
    total_return: float | None
    annualized_return: float | None

    # Convert one performance summary dataclass into dictionary form
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Build the backtest metrics config from the YAML config dictionary
def build_backtest_metrics_config(
    config: dict[str, Any],
    *,
    annualization_factor: int | None = None,
) -> BacktestMetricsConfig:
    resolved_annualization_factor = int(
        annualization_factor
        if annualization_factor is not None
        else _get_config_value(
            config,
            ["backtest", "annualization_factor"],
            _get_config_value(config, ["split", "trading_days"], 252),
        )
    )
    return BacktestMetricsConfig(
        rf_rate_daily = float(_get_config_value(config, ["metrics", "rf_rate_daily"], 0.0)),
        annualization_factor = resolved_annualization_factor,
        initial_equity = float(_get_config_value(config, ["metrics", "initial_equity"], 1.0)),
    )


# Build the daily realized PnL and equity curve from the trade ledger
def build_equity_curve(
    trade_ledger: pd.DataFrame,
    trading_window: pd.DataFrame,
    *,
    strategy: str | None = None,
    initial_equity: float = 1.0,
) -> pd.DataFrame:
    resolved_window = validate_trading_window(trading_window)
    resolved_trade_ledger = _normalize_trade_ledger(trade_ledger)
    processed_window = resolved_window.reset_index(drop = True).copy()

    equity_curve = processed_window.copy()
    equity_curve.insert(0, "day_index", np.arange(len(processed_window), dtype = np.int32))
    equity_curve["strategy"] = strategy if strategy is not None else _infer_strategy(resolved_trade_ledger)
    equity_curve["daily_net_pnl"] = 0.0
    equity_curve["daily_trade_count"] = 0
    equity_curve["daily_forced_exit_count"] = 0

    if not resolved_trade_ledger.empty:
        if (resolved_trade_ledger["exit_idx"] < 0).any() or (
            resolved_trade_ledger["exit_idx"] >= len(processed_window)
        ).any():
            raise ValueError("Trade ledger contains exit_idx values outside the trading window.")

        by_exit_idx = resolved_trade_ledger.groupby("exit_idx", sort = True)
        equity_curve.loc[by_exit_idx["net_pnl"].sum().index, "daily_net_pnl"] = (
            by_exit_idx["net_pnl"].sum().to_numpy(dtype = np.float64)
        )
        equity_curve.loc[by_exit_idx.size().index, "daily_trade_count"] = (
            by_exit_idx.size().to_numpy(dtype = np.int32)
        )
        if "forced_exit" in resolved_trade_ledger.columns:
            forced_by_exit_idx = by_exit_idx["forced_exit"].sum()
            equity_curve.loc[forced_by_exit_idx.index, "daily_forced_exit_count"] = (
                forced_by_exit_idx.to_numpy(dtype = np.int32)
            )

    equity_curve["cumulative_net_pnl"] = equity_curve["daily_net_pnl"].cumsum()
    equity_curve["equity"] = float(initial_equity) + equity_curve["cumulative_net_pnl"]
    equity_curve["daily_return"] = equity_curve["equity"].pct_change(fill_method = None).fillna(0.0)
    equity_curve["cumulative_trade_count"] = equity_curve["daily_trade_count"].cumsum()
    equity_curve["running_peak"] = equity_curve["equity"].cummax()
    equity_curve["drawdown"] = equity_curve["equity"] / equity_curve["running_peak"] - 1.0
    return equity_curve


# Compute the annualized Sharpe ratio from daily returns
def compute_annualized_sharpe(
    daily_returns: pd.Series,
    *,
    rf_rate_daily: float = 0.0,
    annualization_factor: int = 252,
) -> float | None:
    resolved_returns = pd.Series(daily_returns, dtype = np.float64)
    if resolved_returns.empty:
        return None

    excess_returns = resolved_returns - float(rf_rate_daily)
    return_std = float(excess_returns.std(ddof = 1))
    if not np.isfinite(return_std) or return_std <= 0.0:
        return None

    return float(np.sqrt(float(annualization_factor)) * excess_returns.mean() / return_std)


# Compute the maximum drawdown from an equity curve
def compute_max_drawdown(equity_curve: pd.DataFrame) -> float:
    if equity_curve.empty:
        return 0.0
    if "drawdown" not in equity_curve.columns:
        raise ValueError("Equity curve must contain a 'drawdown' column.")
    return float(pd.Series(equity_curve["drawdown"], dtype = np.float64).min())


# Build the summary performance metrics for one strategy run
def build_performance_summary(
    trade_ledger: pd.DataFrame,
    equity_curve: pd.DataFrame,
    *,
    metrics_config: BacktestMetricsConfig,
    strategy: str | None = None,
    pair: str | None = None,
) -> BacktestPerformanceSummary:
    resolved_trade_ledger = _normalize_trade_ledger(trade_ledger)
    resolved_equity_curve = _normalize_equity_curve(equity_curve)

    trade_count = int(len(resolved_trade_ledger))
    win_rate = None
    average_holding_days = None
    if trade_count > 0:
        win_rate = float((resolved_trade_ledger["net_pnl"] > 0.0).mean())
        average_holding_days = float(resolved_trade_ledger["holding_days"].mean())

    daily_return_mean = None
    daily_return_std = None
    sharpe_ann = compute_annualized_sharpe(
        resolved_equity_curve["daily_return"],
        rf_rate_daily = metrics_config.rf_rate_daily,
        annualization_factor = metrics_config.annualization_factor,
    )
    if not resolved_equity_curve.empty:
        daily_return_mean = float(resolved_equity_curve["daily_return"].mean())
        daily_return_std = float(resolved_equity_curve["daily_return"].std(ddof = 1))
        if not np.isfinite(daily_return_std):
            daily_return_std = None

    total_net_pnl = float(resolved_trade_ledger["net_pnl"].sum()) if trade_count > 0 else 0.0
    final_cumulative_net_pnl = (
        float(resolved_equity_curve["cumulative_net_pnl"].iloc[-1])
        if not resolved_equity_curve.empty
        else 0.0
    )
    final_equity = (
        float(resolved_equity_curve["equity"].iloc[-1])
        if not resolved_equity_curve.empty
        else float(metrics_config.initial_equity)
    )

    total_return = None
    annualized_return = None
    if float(metrics_config.initial_equity) != 0.0:
        total_return = float(final_equity / float(metrics_config.initial_equity) - 1.0)
        if (
            not resolved_equity_curve.empty
            and metrics_config.initial_equity > 0.0
            and final_equity > 0.0
        ):
            annualized_return = float(
                (final_equity / float(metrics_config.initial_equity))
                ** (float(metrics_config.annualization_factor) / float(len(resolved_equity_curve)))
                - 1.0
            )

    return BacktestPerformanceSummary(
        strategy = strategy if strategy is not None else _infer_strategy(resolved_trade_ledger, resolved_equity_curve),
        pair = pair if pair is not None else _infer_pair(resolved_trade_ledger),
        trade_count = trade_count,
        win_rate = win_rate,
        average_holding_days = average_holding_days,
        sharpe_ann = sharpe_ann,
        max_drawdown = compute_max_drawdown(resolved_equity_curve),
        daily_return_mean = daily_return_mean,
        daily_return_std = daily_return_std,
        total_net_pnl = total_net_pnl,
        final_cumulative_net_pnl = final_cumulative_net_pnl,
        final_equity = final_equity,
        total_return = total_return,
        annualized_return = annualized_return,
    )


# Build a comparison table from multiple strategy summaries
def build_comparison_metrics_table(
    summaries: list[BacktestPerformanceSummary],
) -> pd.DataFrame:
    return pd.DataFrame([summary.to_dict() for summary in summaries])


# Save the equity curve to CSV or Parquet
def save_equity_curve(equity_curve: pd.DataFrame, path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)
    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        equity_curve.to_csv(resolved_path, index = False)
    elif suffix == ".parquet":
        equity_curve.to_parquet(resolved_path, index = False)
    else:
        raise ValueError("Equity curve output path must end with .csv or .parquet.")
    return resolved_path


# Save the comparison metrics table to CSV or Parquet
def save_comparison_metrics_table(table: pd.DataFrame, path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)
    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        table.to_csv(resolved_path, index = False)
    elif suffix == ".parquet":
        table.to_parquet(resolved_path, index = False)
    else:
        raise ValueError("Comparison metrics output path must end with .csv or .parquet.")
    return resolved_path


# Validate and normalize the trade ledger input for metrics
def _normalize_trade_ledger(trade_ledger: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(trade_ledger, pd.DataFrame):
        raise TypeError(
            "trade_ledger must be a pandas DataFrame. "
            f"Got: {type(trade_ledger).__name__}"
        )

    required_columns = {"exit_idx", "net_pnl", "holding_days"}
    missing = required_columns.difference(trade_ledger.columns)
    if missing:
        raise ValueError(
            f"trade_ledger is missing required columns: {sorted(missing)}"
        )
    return trade_ledger.reset_index(drop = True).copy()


# Validate and normalize the equity curve input for metrics
def _normalize_equity_curve(equity_curve: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(equity_curve, pd.DataFrame):
        raise TypeError(
            "equity_curve must be a pandas DataFrame. "
            f"Got: {type(equity_curve).__name__}"
        )
    required_columns = {"daily_return", "cumulative_net_pnl", "equity", "drawdown"}
    missing = required_columns.difference(equity_curve.columns)
    if missing:
        raise ValueError(
            f"equity_curve is missing required columns: {sorted(missing)}"
        )
    return equity_curve.reset_index(drop = True).copy()


# Infer the strategy label from available result frames
def _infer_strategy(*frames: pd.DataFrame) -> str | None:
    for frame in frames:
        if isinstance(frame, pd.DataFrame) and "strategy" in frame.columns and not frame.empty:
            return str(frame["strategy"].iloc[0])
    return None


# Infer the traded pair label from the trade ledger
def _infer_pair(trade_ledger: pd.DataFrame) -> str | None:
    if "pair" not in trade_ledger.columns or trade_ledger.empty:
        return None
    value = trade_ledger["pair"].iloc[0]
    return None if pd.isna(value) else str(value)
