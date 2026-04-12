from __future__ import annotations
from dataclasses import dataclass
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from src.sigstop.backtest.accounting import PairTradeAccountingSpec, build_trade_ledger, save_trade_ledger
from src.sigstop.backtest.baseline import BaselineBacktestStrategy
from src.sigstop.backtest.engine import (
    BacktestEngineResult,
    build_backtest_engine_config,
    run_backtest_engine,
    validate_trading_window
)
from src.sigstop.backtest.metrics import (
    BacktestPerformanceSummary,
    build_backtest_metrics_config,
    build_comparison_metrics_table,
    build_equity_curve,
    build_performance_summary,
    save_comparison_metrics_table,
    save_equity_curve
)
from src.sigstop.backtest.reporting import (
    save_drawdown_comparison_plot,
    save_equity_curve_comparison_plot,
    save_holding_time_histogram,
    save_json,
    save_spread_with_trade_markers_plot
)
from src.sigstop.backtest.sot import SOTBacktestStrategy
from src.sigstop.backtest.validation import (
    BacktestValidationResult,
    build_backtest_validation_report,
    save_validation_report
)
from src.sigstop.config import load_config
from src.sigstop.features.manifest import compute_file_sha256, safe_package_version
from src.sigstop.generators.spec import build_generator_context
from src.sigstop.paths import PROCESSED_DATA_DIR, ROOT_DIR, RUNS_DIR, ensure_directories


@dataclass(frozen = True)
class BacktestInputBundle:
    trading_window: pd.DataFrame
    pair_label: str | None
    pair_symbols: list[str]
    beta: float | None
    orientation: str | None
    formation_spread_path: Path | None = None
    trading_spread_path: Path | None = None
    selection_metadata_path: Path | None = None
    split_metadata_path: Path | None = None
    selection_metadata: dict[str, Any] | None = None
    split_metadata: dict[str, Any] | None = None


@dataclass(frozen = True)
class StrategyBacktestArtifacts:
    strategy_name: str
    output_dir: Path
    engine_result: BacktestEngineResult
    trade_ledger: pd.DataFrame
    equity_curve: pd.DataFrame
    summary: BacktestPerformanceSummary
    validation: BacktestValidationResult
    trades_path: Path
    equity_curve_path: Path
    step_records_path: Path
    summary_path: Path
    validation_path: Path


@dataclass(frozen = True)
class FullBacktestRunResult:
    run_id: str
    output_dir: Path
    config_path: Path
    manifest_path: Path
    summary_table_path: Path
    validation_summary_path: Path
    plot_paths: dict[str, Path]
    strategy_runs: dict[str, StrategyBacktestArtifacts]


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Load the default backtest inputs from saved artifacts
def load_backtest_input_bundle(config: dict[str, Any]) -> BacktestInputBundle:
    context = build_generator_context(config)
    selection_metadata = _load_json_file(context.selection_metadata_path)
    split_metadata = _load_json_file(context.split_metadata_path)
    trading_spread = pd.read_parquet(context.trading_spread_path)

    pair_symbols = list(selection_metadata.get("pair", context.pair))
    pair_label = "-".join(pair_symbols) if pair_symbols else None
    beta = selection_metadata.get("chosen_beta")
    orientation = selection_metadata.get("chosen_orientation")
    raw_trading_prices = pd.read_parquet(PROCESSED_DATA_DIR / "trading.parquet")
    merge_columns = ["date"] + [symbol for symbol in pair_symbols if symbol in raw_trading_prices.columns]
    trading_window = trading_spread.merge(
        raw_trading_prices[merge_columns],
        on = "date",
        how = "left",
        validate = "one_to_one",
    )

    return BacktestInputBundle(
        trading_window = validate_trading_window(trading_window),
        pair_label = pair_label,
        pair_symbols = pair_symbols,
        beta = None if beta is None else float(beta),
        orientation = None if orientation is None else str(orientation),
        formation_spread_path = context.formation_spread_path,
        trading_spread_path = context.trading_spread_path,
        selection_metadata_path = context.selection_metadata_path,
        split_metadata_path = context.split_metadata_path,
        selection_metadata = selection_metadata,
        split_metadata = split_metadata,
    )


# Run the full multi-strategy backtest and artifact pipeline
def run_full_backtest(
    config_path: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
    input_bundle: BacktestInputBundle | None = None,
    strategies: dict[str, Any] | None = None,
    run_id: str = "backtest_run",
    output_dir: str | Path | None = None,
) -> FullBacktestRunResult:
    if config is not None and config_path is not None:
        raise ValueError("Provide either config or config_path, not both.")

    resolved_config = (
        dict(config)
        if config is not None
        else (load_config(config_path) if config_path is not None else load_config())
    )
    ensure_directories()

    artifacts_root = Path(_get_config_value(resolved_config, ["artifacts", "root"], RUNS_DIR))
    if not artifacts_root.is_absolute():
        artifacts_root = ROOT_DIR / artifacts_root
    resolved_output_dir = Path(output_dir) if output_dir is not None else artifacts_root / run_id / "backtest"
    resolved_output_dir.mkdir(parents = True, exist_ok = True)

    resolved_input_bundle = input_bundle if input_bundle is not None else load_backtest_input_bundle(resolved_config)
    engine_config = build_backtest_engine_config(resolved_config)
    processed_window = validate_trading_window(resolved_input_bundle.trading_window).iloc[
        : engine_config.trading_days
    ].reset_index(drop = True)

    config_saved_path = save_config_yaml(resolved_config, resolved_output_dir / "config.yaml")
    strategy_suite = strategies or build_default_backtest_strategies(
        config = resolved_config,
        run_id = run_id,
        output_dir = resolved_output_dir,
    )

    strategy_runs: dict[str, StrategyBacktestArtifacts] = {}
    for strategy_name, strategy in strategy_suite.items():
        strategy_output_dir = resolved_output_dir / strategy_name
        strategy_runs[strategy_name] = run_strategy_backtest(
            strategy_name = strategy_name,
            strategy = strategy,
            trading_window = processed_window,
            config = resolved_config,
            pair_label = resolved_input_bundle.pair_label,
            beta = resolved_input_bundle.beta,
            orientation = resolved_input_bundle.orientation,
            output_dir = strategy_output_dir,
        )

    summary_table = build_comparison_metrics_table(
        [strategy_run.summary for strategy_run in strategy_runs.values()]
    )
    summary_table_path = save_comparison_metrics_table(
        summary_table,
        resolved_output_dir / "summary_table.csv",
    )

    validation_summary = {
        strategy_name: strategy_run.validation.to_dict()
        for strategy_name, strategy_run in strategy_runs.items()
    }
    validation_summary_path = save_json(
        validation_summary,
        resolved_output_dir / "validation_summary.json",
    )

    plot_paths = build_backtest_plots(
        trading_window = processed_window,
        strategy_runs = strategy_runs,
        output_dir = resolved_output_dir / "plots",
        save_plots = bool(_get_config_value(resolved_config, ["artifacts", "save_plots"], True)),
    )

    manifest_path = save_json(
        build_backtest_manifest(
            run_id = run_id,
            output_dir = resolved_output_dir,
            config_saved_path = config_saved_path,
            config = resolved_config,
            input_bundle = resolved_input_bundle,
            strategy_suite = strategy_suite,
            strategy_runs = strategy_runs,
            summary_table_path = summary_table_path,
            validation_summary_path = validation_summary_path,
            plot_paths = plot_paths,
        ),
        resolved_output_dir / "manifest.json",
    )

    return FullBacktestRunResult(
        run_id = run_id,
        output_dir = resolved_output_dir,
        config_path = config_saved_path,
        manifest_path = manifest_path,
        summary_table_path = summary_table_path,
        validation_summary_path = validation_summary_path,
        plot_paths = plot_paths,
        strategy_runs = strategy_runs,
    )


# Build the default SOT and baseline strategy suite for one run
def build_default_backtest_strategies(
    *,
    config: dict[str, Any],
    run_id: str,
    output_dir: Path,
) -> dict[str, Any]:
    return {
        "sot": SOTBacktestStrategy(
            config = config,
            output_root = output_dir / "sot",
            run_id = run_id,
        ),
        "baseline": BaselineBacktestStrategy(config = config),
    }


# Run one strategy through the engine, accounting, metrics, and validation stack
def run_strategy_backtest(
    *,
    strategy_name: str,
    strategy: Any,
    trading_window: pd.DataFrame,
    config: dict[str, Any],
    pair_label: str | None,
    beta: float | None,
    orientation: str | None,
    output_dir: Path,
) -> StrategyBacktestArtifacts:
    output_dir.mkdir(parents = True, exist_ok = True)
    metrics_config = build_backtest_metrics_config(
        config,
        annualization_factor = build_backtest_engine_config(config).annualization_factor,
    )
    pair_trade_accounting = _build_pair_trade_accounting_spec(
        strategy_name = strategy_name,
        beta = beta,
        orientation = orientation,
    )

    engine_result = run_backtest_engine(trading_window, strategy, config = config)
    step_records_frame = build_step_records_frame(engine_result)
    step_records_path = save_dataframe(step_records_frame, output_dir / "step_records.csv")

    ledger_result = build_trade_ledger(
        engine_result,
        trading_window,
        config = config,
        pair = pair_label,
        beta = beta,
        initial_equity = metrics_config.initial_equity,
        pair_trade_accounting = pair_trade_accounting,
    )
    trades_path = save_trade_ledger(ledger_result.trade_ledger, output_dir / "trades.csv")

    metrics_config = build_backtest_metrics_config(
        config,
        annualization_factor = engine_result.engine_config.annualization_factor,
    )
    equity_curve = build_equity_curve(
        ledger_result.trade_ledger,
        trading_window.iloc[: engine_result.trading_window_length].reset_index(drop = True),
        strategy = strategy_name,
        initial_equity = metrics_config.initial_equity,
        pair_trade_accounting = pair_trade_accounting,
    )
    equity_curve_path = save_equity_curve(equity_curve, output_dir / "equity_curve.csv")

    summary = build_performance_summary(
        ledger_result.trade_ledger,
        equity_curve,
        metrics_config = metrics_config,
        strategy = strategy_name,
        pair = pair_label,
    )
    summary_path = save_json(summary.to_dict(), output_dir / "summary.json")

    validation = build_backtest_validation_report(
        strategy = strategy_name,
        engine_result = engine_result,
        trade_ledger = ledger_result.trade_ledger,
        trading_window = trading_window.iloc[: engine_result.trading_window_length].reset_index(drop = True),
        indicator_frame = getattr(strategy, "indicator_frame", None),
        baseline_strategy_config = getattr(strategy, "strategy_config", None)
        if strategy_name == "baseline"
        else None,
    )
    validation_path = save_validation_report(validation, output_dir / "validation.json")

    return StrategyBacktestArtifacts(
        strategy_name = strategy_name,
        output_dir = output_dir,
        engine_result = engine_result,
        trade_ledger = ledger_result.trade_ledger,
        equity_curve = equity_curve,
        summary = summary,
        validation = validation,
        trades_path = trades_path,
        equity_curve_path = equity_curve_path,
        step_records_path = step_records_path,
        summary_path = summary_path,
        validation_path = validation_path,
    )


def _build_pair_trade_accounting_spec(
    *,
    strategy_name: str,
    beta: float | None,
    orientation: str | None,
) -> PairTradeAccountingSpec | None:
    if strategy_name != "baseline" or beta is None or orientation is None:
        return None
    if orientation == "GS_minus_beta_MS":
        leg_1_symbol = "GS"
        leg_2_symbol = "MS"
    elif orientation == "MS_minus_beta_GS":
        leg_1_symbol = "MS"
        leg_2_symbol = "GS"
    else:
        raise ValueError(f"Unsupported spread orientation for pair accounting: {orientation!r}")

    return PairTradeAccountingSpec(
        leg_1_symbol = leg_1_symbol,
        leg_2_symbol = leg_2_symbol,
        beta = float(beta),
    )


# Convert engine step records into a tabular artifact
def build_step_records_frame(engine_result: BacktestEngineResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in engine_result.step_records:
        rows.append(
            {
                "day_index": int(record.day_index),
                "date": record.date,
                "spread": float(record.spread),
                "hook_name": record.hook_name,
                "state_before": record.state_before.value,
                "action": record.action.value,
                "state_after": record.state_after.value,
                "state_start_index_before": int(record.state_start_index_before),
                "state_start_index_after": int(record.state_start_index_after),
                "position_entry_index": record.position_entry_index,
                "decision_metadata_json": json.dumps(record.decision_metadata, sort_keys = True, ensure_ascii = True),
            }
        )
    return pd.DataFrame(rows)


# Build and save the standard comparison plots for the backtest run
def build_backtest_plots(
    *,
    trading_window: pd.DataFrame,
    strategy_runs: dict[str, StrategyBacktestArtifacts],
    output_dir: Path,
    save_plots: bool,
) -> dict[str, Path]:
    if not save_plots:
        return {}
    output_dir.mkdir(parents = True, exist_ok = True)

    equity_curves = {
        strategy_name: strategy_run.equity_curve
        for strategy_name, strategy_run in strategy_runs.items()
    }
    trade_ledgers = {
        strategy_name: strategy_run.trade_ledger
        for strategy_name, strategy_run in strategy_runs.items()
    }

    return {
        "equity_curve_comparison": save_equity_curve_comparison_plot(
            equity_curves,
            output_dir / "equity_curve_comparison.png",
        ),
        "drawdown_comparison": save_drawdown_comparison_plot(
            equity_curves,
            output_dir / "drawdown_comparison.png",
        ),
        "holding_time_histogram": save_holding_time_histogram(
            trade_ledgers,
            output_dir / "holding_time_histogram.png",
        ),
        "spread_with_trade_markers": save_spread_with_trade_markers_plot(
            trading_window,
            trade_ledgers,
            output_dir / "spread_with_trade_markers.png",
        ),
    }


# Build the reproducibility manifest for the full backtest run
def build_backtest_manifest(
    *,
    run_id: str,
    output_dir: Path,
    config_saved_path: Path,
    config: dict[str, Any],
    input_bundle: BacktestInputBundle,
    strategy_suite: dict[str, Any],
    strategy_runs: dict[str, StrategyBacktestArtifacts],
    summary_table_path: Path,
    validation_summary_path: Path,
    plot_paths: dict[str, Path],
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "output_dir": str(output_dir),
        "config_path": str(config_saved_path),
        "config_hash": compute_file_sha256(config_saved_path),
        "inputs": {
            "pair_label": input_bundle.pair_label,
            "pair_symbols": list(input_bundle.pair_symbols),
            "beta": input_bundle.beta,
            "orientation": input_bundle.orientation,
            "trading_spread_path": None if input_bundle.trading_spread_path is None else str(input_bundle.trading_spread_path),
            "trading_spread_hash": _hash_optional_file(input_bundle.trading_spread_path),
            "formation_spread_path": None if input_bundle.formation_spread_path is None else str(input_bundle.formation_spread_path),
            "formation_spread_hash": _hash_optional_file(input_bundle.formation_spread_path),
            "selection_metadata_path": None if input_bundle.selection_metadata_path is None else str(input_bundle.selection_metadata_path),
            "selection_metadata_hash": _hash_optional_file(input_bundle.selection_metadata_path),
            "split_metadata_path": None if input_bundle.split_metadata_path is None else str(input_bundle.split_metadata_path),
            "split_metadata_hash": _hash_optional_file(input_bundle.split_metadata_path),
            "selection_metadata": input_bundle.selection_metadata,
            "split_metadata": input_bundle.split_metadata,
        },
        "repro": {
            "seed": _get_config_value(config, ["repro", "seed"], None),
            "git_commit": resolve_git_commit(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "os_name": os.name,
            "versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "matplotlib": safe_package_version("matplotlib"),
                "torch": safe_package_version("torch"),
                "pyyaml": safe_package_version("PyYAML"),
            },
        },
        "strategies": {
            strategy_name: build_strategy_manifest_entry(
                strategy_name = strategy_name,
                strategy = strategy_suite[strategy_name],
                artifacts = strategy_run,
            )
            for strategy_name, strategy_run in strategy_runs.items()
        },
        "outputs": {
            "summary_table_path": str(summary_table_path),
            "summary_table_hash": compute_file_sha256(summary_table_path),
            "validation_summary_path": str(validation_summary_path),
            "validation_summary_hash": compute_file_sha256(validation_summary_path),
            "plots": {
                name: {
                    "path": str(path),
                    "hash": compute_file_sha256(path),
                }
                for name, path in plot_paths.items()
            },
        },
    }


# Build the manifest entry for one strategy's backtest artifacts
def build_strategy_manifest_entry(
    *,
    strategy_name: str,
    strategy: Any,
    artifacts: StrategyBacktestArtifacts,
) -> dict[str, Any]:
    return {
        "strategy_name": strategy_name,
        "output_dir": str(artifacts.output_dir),
        "trade_count": int(len(artifacts.trade_ledger)),
        "summary": artifacts.summary.to_dict(),
        "validation": artifacts.validation.to_dict(),
        "artifacts": {
            "trades_path": str(artifacts.trades_path),
            "trades_hash": compute_file_sha256(artifacts.trades_path),
            "equity_curve_path": str(artifacts.equity_curve_path),
            "equity_curve_hash": compute_file_sha256(artifacts.equity_curve_path),
            "step_records_path": str(artifacts.step_records_path),
            "step_records_hash": compute_file_sha256(artifacts.step_records_path),
            "summary_path": str(artifacts.summary_path),
            "summary_hash": compute_file_sha256(artifacts.summary_path),
            "validation_path": str(artifacts.validation_path),
            "validation_hash": compute_file_sha256(artifacts.validation_path),
        },
        "model_artifact_references": collect_strategy_model_references(strategy),
    }


# Collect model artifact references from a strategy's stage history
def collect_strategy_model_references(strategy: Any) -> list[dict[str, Any]]:
    stage_histories = []
    for attribute in ("entry_stage_history", "exit_stage_history"):
        history = getattr(strategy, attribute, None)
        if history is not None:
            stage_histories.extend(list(history))

    references: list[dict[str, Any]] = []
    for plan in stage_histories:
        references.append(
            {
                "stage": getattr(plan, "stage", None),
                "state_start_index": getattr(plan, "state_start_index", None),
                "policy_id": getattr(plan, "policy_id", None),
                "threshold": getattr(plan, "threshold", None),
                "horizon": getattr(plan, "horizon", None),
                "output_dir": None if getattr(plan, "output_dir", None) is None else str(getattr(plan, "output_dir")),
                "feature_cache": {
                    "cache_key": getattr(plan, "feature_cache_key", None),
                    "cache_hit": bool(getattr(plan, "feature_cache_hit", False)),
                    "feature_path": None
                    if getattr(plan, "feature_cache_path", None) is None
                    else str(getattr(plan, "feature_cache_path")),
                    "feature_path_hash": _hash_optional_file(getattr(plan, "feature_cache_path", None)),
                    "manifest_path": None
                    if getattr(plan, "feature_manifest_path", None) is None
                    else str(getattr(plan, "feature_manifest_path")),
                    "manifest_path_hash": _hash_optional_file(getattr(plan, "feature_manifest_path", None)),
                },
                "training_artifacts": collect_training_artifacts(
                    getattr(getattr(plan, "training_result", None), "artifacts", None)
                ),
            }
        )
    return references


# Collect serializable training artifact paths and hashes
def collect_training_artifacts(training_artifacts: Any) -> dict[str, Any] | None:
    if training_artifacts is None:
        return None

    result: dict[str, Any] = {}
    for attribute in (
        "output_dir",
        "checkpoint_last_path",
        "checkpoint_best_path",
        "policy_best_path",
        "optimizer_last_path",
        "optimizer_best_path",
        "history_path",
        "manifest_path",
    ):
        if not hasattr(training_artifacts, attribute):
            continue
        value = getattr(training_artifacts, attribute)
        if value is None:
            result[attribute] = None
            continue
        resolved = Path(value)
        result[attribute] = str(resolved)
        if resolved.exists() and resolved.is_file():
            result[f"{attribute}_hash"] = compute_file_sha256(resolved)
    return result


# Save the resolved backtest config to YAML
def save_config_yaml(config: dict[str, Any], path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)
    with open(resolved_path, "w", encoding = "utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys = False)
    return resolved_path


# Save a DataFrame artifact to CSV or Parquet
def save_dataframe(frame: pd.DataFrame, path: str | Path) -> Path:
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents = True, exist_ok = True)
    suffix = resolved_path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(resolved_path, index = False)
    elif suffix == ".parquet":
        frame.to_parquet(resolved_path, index = False)
    else:
        raise ValueError("Output path must end with .csv or .parquet.")
    return resolved_path


# Resolve the current git commit hash when the repo is available
def resolve_git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd = ROOT_DIR,
            check = True,
            capture_output = True,
            text = True,
            timeout = 5,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


# Hash a file path only when the file exists
def _hash_optional_file(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = Path(path)
    if not resolved.exists() or not resolved.is_file():
        return None
    return compute_file_sha256(resolved)


# Load a JSON artifact from disk
def _load_json_file(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding = "utf-8") as handle:
        return json.load(handle)
