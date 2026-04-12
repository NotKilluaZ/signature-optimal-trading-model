from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.sigstop.backtest.baseline import BaselineBacktestStrategy
from src.sigstop.backtest.runner import BacktestInputBundle, run_full_backtest
from src.sigstop.backtest.sot import SOTBacktestStrategy
from src.sigstop.features.feature_builder import FeatureBuildResult
from src.sigstop.generators.types import OUGeneratorParams
from src.sigstop.stopping.policy import LinearStoppingPolicy


def _toy_config() -> dict:
    return {
        "backtest": {
            "trading_days": 6,
            "execution_price": "close",
            "allow_forced_exit_end": True,
            "annualization_factor": 252,
        },
        "artifacts": {
            "root": "runs",
            "save_trades": True,
            "save_equity_curve": True,
            "save_plots": True,
            "save_manifests": True,
        },
        "costs": {
            "fixed_per_action": 0.0,
            "proportional": 0.0,
            "slippage_bps": 0.0,
            "model": "spread_proxy",
        },
        "metrics": {
            "rf_rate_daily": 0.0,
            "initial_equity": 1.0,
        },
        "baseline": {
            "ma_window": 3,
            "k": 0.1,
            "min_periods": 3,
            "use_shift_1": True,
        },
        "split": {
            "trading_days": 6,
        },
        "features": {
            "signature": {
                "depth": 2,
                "scalar_term": True,
                "min_prefix": 2,
                "dtype": "float32",
                "library": "esig",
                "device": "cpu",
                "basepoint": False,
            },
            "augmentation": {
                "time_representation": "index_0_1",
            },
        },
        "stopping": {
            "deterministic_threshold": {
                "k": 0.99,
                "mu": 5.0,
            },
        },
        "training": {
            "M": 9,
        },
        "generator": {
            "fit": {
                "dt": 1.0,
            },
            "sample": {
                "n_paths": 11,
                "dtype": "float64",
                "device": "cpu",
                "include_innovations": False,
            },
        },
        "repro": {
            "seed": 7,
        },
        "sot": {
            "k": 0.05,
            "mu": 20.0,
            "ou_samples_M": 3,
            "feature_depth_N": 1,
            "retrain_cadence": "per_trade",
            "cache_episode_features": True,
        },
    }


def _toy_window(n_days: int = 6) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2022-01-03", periods = n_days, freq = "B"),
            "spread": [10.0, 10.0, 10.0, 9.0, 10.0, 10.0][:n_days],
            "GS": [20.0, 20.0, 20.0, 19.0, 20.0, 20.0][:n_days],
            "MS": [10.0, 10.0, 10.0, 10.0, 10.0, 10.0][:n_days],
        }
    )


def _toy_ou_params() -> OUGeneratorParams:
    return OUGeneratorParams(
        kappa = 0.5,
        theta = 0.0,
        sigma = 1.0,
        dt = 1.0,
        phi = float(np.exp(-0.5)),
        half_life_days = 1.0,
        estimation_method = "test",
    )


def test_run_full_backtest_writes_artifacts_manifest_and_plots(monkeypatch) -> None:
    def fake_sample_stage_paths(config, ou_params, *, x0, horizon):
        paths = np.tile(
            np.linspace(float(x0), float(x0) + float(max(horizon, 0)), int(horizon) + 1, dtype = np.float64),
            (int(config["generator"]["sample"]["n_paths"]), 1),
        )
        return paths, SimpleNamespace(horizon = int(horizon), x0 = float(x0))

    def fake_build_real_stage_feature_tensor(spread_segment, formation_spread, config):
        segment = np.asarray(spread_segment, dtype = np.float64)
        n_prefix = max(len(segment) - 1, 0)
        return FeatureBuildResult(
            features = np.ones((n_prefix, 1), dtype = np.float32),
            scaled_spread = segment.astype(np.float32, copy = False),
            augmented_path = np.zeros((len(segment), 2), dtype = np.float32),
            prefix_ends = np.arange(1, len(segment), dtype = np.int32),
            feature_spec = {"n_prefix": n_prefix},
        )

    def fake_train_entry_policy(
        config_path = None,
        *,
        config = None,
        output_dir = None,
        run_id = None,
        spread_paths = None,
        formation_spread = None,
        resume_from = None,
        cache_base_dir = None,
        cache_source = None,
        extra_metadata = None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents = True, exist_ok = True)
        manifest_path = output_dir / "manifest.json"
        policy_best_path = output_dir / "policy_best.pt"
        manifest_path.write_text("{\"stage\":\"entry\"}", encoding = "utf-8")
        policy_best_path.write_text("entry-policy", encoding = "utf-8")
        return SimpleNamespace(
            best_policy = LinearStoppingPolicy(weights = np.array([1.0], dtype = np.float32)),
            artifacts = SimpleNamespace(
                output_dir = output_dir,
                manifest_path = manifest_path,
                policy_best_path = policy_best_path,
            ),
        )

    def fake_train_exit_policy(
        config_path = None,
        *,
        config = None,
        output_dir = None,
        run_id = None,
        spread_paths = None,
        formation_spread = None,
        resume_from = None,
        cache_base_dir = None,
        cache_source = None,
        extra_metadata = None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents = True, exist_ok = True)
        manifest_path = output_dir / "manifest.json"
        policy_best_path = output_dir / "policy_best.pt"
        manifest_path.write_text("{\"stage\":\"exit\"}", encoding = "utf-8")
        policy_best_path.write_text("exit-policy", encoding = "utf-8")
        return SimpleNamespace(
            best_policy = LinearStoppingPolicy(weights = np.array([1.0], dtype = np.float32)),
            artifacts = SimpleNamespace(
                output_dir = output_dir,
                manifest_path = manifest_path,
                policy_best_path = policy_best_path,
            ),
        )

    monkeypatch.setattr("src.sigstop.backtest.sot.sample_sot_stage_paths", fake_sample_stage_paths)
    monkeypatch.setattr("src.sigstop.backtest.sot.build_real_stage_feature_tensor", fake_build_real_stage_feature_tensor)
    monkeypatch.setattr("src.sigstop.backtest.sot.train_entry_policy", fake_train_entry_policy)
    monkeypatch.setattr("src.sigstop.backtest.sot.train_exit_policy", fake_train_exit_policy)

    config = _toy_config()
    config["sot"]["cache_episode_features"] = False
    output_dir = Path("runs") / "pytest_artifacts" / "full_backtest_run"
    strategies = {
        "sot": SOTBacktestStrategy(
            config = config,
            formation_spread = np.array([9.0, 10.0, 11.0, 12.0], dtype = np.float64),
            ou_params = _toy_ou_params(),
            output_root = output_dir / "sot",
            run_id = "pytest_full_backtest",
        ),
        "baseline": BaselineBacktestStrategy(config = config),
    }
    result = run_full_backtest(
        config = config,
        input_bundle = BacktestInputBundle(
            trading_window = _toy_window(),
            pair_label = "GS-MS",
            pair_symbols = ["GS", "MS"],
            beta = 1.0,
            orientation = "GS_minus_beta_MS",
            selection_metadata = {"pair": ["GS", "MS"], "chosen_beta": 1.0},
            split_metadata = {"trading_days": 6},
        ),
        strategies = strategies,
        run_id = "pytest_full_backtest",
        output_dir = output_dir,
    )

    assert result.config_path.exists()
    assert result.manifest_path.exists()
    assert result.summary_table_path.exists()
    assert result.validation_summary_path.exists()
    assert set(result.plot_paths) == {
        "equity_curve_comparison",
        "drawdown_comparison",
        "holding_time_histogram",
        "spread_with_trade_markers",
    }
    for path in result.plot_paths.values():
        assert path.exists()

    assert (output_dir / "sot" / "trades.csv").exists()
    assert (output_dir / "sot" / "equity_curve.csv").exists()
    assert (output_dir / "baseline" / "trades.csv").exists()
    assert (output_dir / "baseline" / "equity_curve.csv").exists()

    manifest = json.loads(result.manifest_path.read_text(encoding = "utf-8"))
    assert manifest["run_id"] == "pytest_full_backtest"
    assert manifest["inputs"]["pair_label"] == "GS-MS"
    assert manifest["config_hash"]
    assert "numpy" in manifest["repro"]["versions"]
    assert manifest["outputs"]["summary_table_hash"]
    assert manifest["strategies"]["sot"]["artifacts"]["trades_hash"]
    assert manifest["strategies"]["baseline"]["artifacts"]["equity_curve_hash"]
    assert len(manifest["strategies"]["sot"]["model_artifact_references"]) >= 1
    assert manifest["strategies"]["sot"]["validation"]["passed"] is True
    assert manifest["strategies"]["baseline"]["validation"]["passed"] is True
