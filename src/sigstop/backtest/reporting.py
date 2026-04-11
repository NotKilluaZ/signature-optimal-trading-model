from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

plt.switch_backend("Agg")


# Save a JSON artifact to disk
def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    import json

    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)
    with open(resolved_path, "w", encoding = "utf-8") as handle:
        json.dump(payload, handle, indent = 2)
    return resolved_path


# Save a comparison plot of strategy equity curves
def save_equity_curve_comparison_plot(
    equity_curves: dict[str, pd.DataFrame],
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (9, 4.8))
    for strategy_name, frame in equity_curves.items():
        x = frame["date"] if "date" in frame.columns else frame["day_index"]
        plt.plot(x, frame["equity"], label = strategy_name, linewidth = 2.0)
    plt.title("Equity Curve Comparison")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolved_path)
    plt.close()
    return resolved_path


# Save a comparison plot of strategy drawdown curves
def save_drawdown_comparison_plot(
    equity_curves: dict[str, pd.DataFrame],
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (9, 4.8))
    for strategy_name, frame in equity_curves.items():
        x = frame["date"] if "date" in frame.columns else frame["day_index"]
        plt.plot(x, frame["drawdown"], label = strategy_name, linewidth = 2.0)
    plt.title("Drawdown Comparison")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.tight_layout()
    plt.savefig(resolved_path)
    plt.close()
    return resolved_path


# Save a comparison histogram of trade holding times
def save_holding_time_histogram(
    trade_ledgers: dict[str, pd.DataFrame],
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (9, 4.8))
    has_any_data = False
    for strategy_name, ledger in trade_ledgers.items():
        if "holding_days" not in ledger.columns or ledger.empty:
            continue
        has_any_data = True
        plt.hist(
            ledger["holding_days"].to_numpy(dtype = float),
            bins = min(max(len(ledger), 1), 20),
            alpha = 0.55,
            label = strategy_name,
        )
    plt.title("Holding Time Distribution")
    plt.xlabel("Holding Days")
    plt.ylabel("Trade Count")
    if has_any_data:
        plt.legend()
    plt.tight_layout()
    plt.savefig(resolved_path)
    plt.close()
    return resolved_path


# Save the spread series with strategy entry and exit markers
def save_spread_with_trade_markers_plot(
    trading_window: pd.DataFrame,
    trade_ledgers: dict[str, pd.DataFrame],
    path: str | Path,
) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)

    plt.figure(figsize = (10, 4.8))
    x = trading_window["date"] if "date" in trading_window.columns else trading_window.index
    plt.plot(x, trading_window["spread"], color = "black", linewidth = 1.5, label = "spread")

    colors = {
        "sot": "tab:blue",
        "baseline": "tab:orange",
    }
    for strategy_name, ledger in trade_ledgers.items():
        if ledger.empty:
            continue
        color = colors.get(strategy_name, None)
        entry_x = (
            trading_window.iloc[ledger["entry_idx"].to_numpy(dtype = int)]["date"]
            if "date" in trading_window.columns
            else ledger["entry_idx"]
        )
        exit_x = (
            trading_window.iloc[ledger["exit_idx"].to_numpy(dtype = int)]["date"]
            if "date" in trading_window.columns
            else ledger["exit_idx"]
        )
        plt.scatter(
            entry_x,
            ledger["entry_spread"],
            marker = "^",
            s = 45,
            color = color,
            label = f"{strategy_name} entry",
        )
        plt.scatter(
            exit_x,
            ledger["exit_spread"],
            marker = "v",
            s = 45,
            color = color,
            facecolors = "none",
            label = f"{strategy_name} exit",
        )

    handles, labels = plt.gca().get_legend_handles_labels()
    unique: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    plt.title("Spread With Trade Markers")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.legend(unique.values(), unique.keys(), fontsize = 8)
    plt.tight_layout()
    plt.savefig(resolved_path)
    plt.close()
    return resolved_path
