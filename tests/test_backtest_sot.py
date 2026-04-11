from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pandas as pd

from src.sigstop.backtest.engine import BacktestEngineConfig, StrategyAction, run_backtest_engine
from src.sigstop.backtest.sot import SOTBacktestStrategy, build_effective_sot_config
from src.sigstop.features.feature_builder import FeatureBuildResult
from src.sigstop.generators.types import OUGeneratorParams
from src.sigstop.stopping.policy import LinearStoppingPolicy


def _toy_config() -> dict:
    return {
        "split": {
            "trading_days": 252,
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
            "spread": [10.0 + float(index) for index in range(n_days)],
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


def test_build_effective_sot_config_overlays_strategy_defaults_onto_training_inputs() -> None:
    config = build_effective_sot_config(_toy_config())

    assert config["sot"]["k"] == 0.05
    assert config["stopping"]["deterministic_threshold"]["k"] == 0.05
    assert config["sot"]["mu"] == 20.0
    assert config["stopping"]["deterministic_threshold"]["mu"] == 20.0
    assert config["sot"]["ou_samples_M"] == 3
    assert config["training"]["M"] == 3
    assert config["generator"]["sample"]["n_paths"] == 3
    assert config["sot"]["feature_depth_N"] == 1
    assert config["features"]["signature"]["depth"] == 1


def test_sot_strategy_retrains_once_per_trade_stage_and_repeats_inside_engine(monkeypatch) -> None:
    entry_training_calls: list[dict[str, object]] = []
    exit_training_calls: list[dict[str, object]] = []

    def fake_sample_stage_paths(config, ou_params, *, x0, horizon):
        paths = np.tile(
            np.linspace(float(x0), float(x0) + float(max(horizon, 0)), int(horizon) + 1, dtype = np.float64),
            (int(config["generator"]["sample"]["n_paths"]), 1),
        )
        request = SimpleNamespace(horizon = int(horizon), x0 = float(x0))
        return paths, request

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
        entry_training_calls.append(
            {
                "x0": float(spread_paths[0, 0]),
                "horizon": int(spread_paths.shape[1] - 1),
                "output_dir": str(output_dir),
                "n_paths": int(spread_paths.shape[0]),
                "cache_base_dir": None if cache_base_dir is None else str(cache_base_dir),
                "cache_source_stage": None if cache_source is None else cache_source.get("stage"),
            }
        )
        return SimpleNamespace(
            best_policy = LinearStoppingPolicy(weights = np.array([1.0], dtype = np.float32)),
            artifacts = SimpleNamespace(output_dir = Path(output_dir)),
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
        exit_training_calls.append(
            {
                "x0": float(spread_paths[0, 0]),
                "horizon": int(spread_paths.shape[1] - 1),
                "output_dir": str(output_dir),
                "n_paths": int(spread_paths.shape[0]),
                "cache_base_dir": None if cache_base_dir is None else str(cache_base_dir),
                "cache_source_stage": None if cache_source is None else cache_source.get("stage"),
            }
        )
        return SimpleNamespace(
            best_policy = LinearStoppingPolicy(weights = np.array([1.0], dtype = np.float32)),
            artifacts = SimpleNamespace(output_dir = Path(output_dir)),
        )

    monkeypatch.setattr("src.sigstop.backtest.sot.sample_sot_stage_paths", fake_sample_stage_paths)
    monkeypatch.setattr("src.sigstop.backtest.sot.build_real_stage_feature_tensor", fake_build_real_stage_feature_tensor)
    monkeypatch.setattr("src.sigstop.backtest.sot.train_entry_policy", fake_train_entry_policy)
    monkeypatch.setattr("src.sigstop.backtest.sot.train_exit_policy", fake_train_exit_policy)

    config = _toy_config()
    config["sot"]["cache_episode_features"] = False
    strategy = SOTBacktestStrategy(
        config = config,
        formation_spread = np.array([9.0, 10.0, 11.0, 12.0], dtype = np.float64),
        ou_params = _toy_ou_params(),
        output_root = Path("runs") / "pytest_artifacts" / "sot_strategy_retrain",
        run_id = "pytest_sot",
    )
    result = run_backtest_engine(
        _toy_window(n_days = 6),
        strategy,
        engine_config = BacktestEngineConfig(trading_days = 6),
    )

    assert [record.action for record in result.step_records] == [
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
        StrategyAction.EXIT_LONG_SPREAD,
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
        StrategyAction.EXIT_LONG_SPREAD,
    ]

    assert len(entry_training_calls) == 2
    assert len(exit_training_calls) == 2

    expected_training_cache_root = str((Path.cwd() / "data" / "synthetic" / "gs_ms" / "ou").resolve())

    assert entry_training_calls == [
        {
            "x0": 10.0,
            "horizon": 5,
            "output_dir": str(Path("runs") / "pytest_artifacts" / "sot_strategy_retrain" / "entry" / "entry_start_0000"),
            "n_paths": 3,
            "cache_base_dir": expected_training_cache_root,
            "cache_source_stage": "entry",
        },
        {
            "x0": 13.0,
            "horizon": 2,
            "output_dir": str(Path("runs") / "pytest_artifacts" / "sot_strategy_retrain" / "entry" / "entry_start_0003"),
            "n_paths": 3,
            "cache_base_dir": expected_training_cache_root,
            "cache_source_stage": "entry",
        },
    ]
    assert exit_training_calls == [
        {
            "x0": 11.0,
            "horizon": 4,
            "output_dir": str(Path("runs") / "pytest_artifacts" / "sot_strategy_retrain" / "exit" / "exit_start_0001"),
            "n_paths": 3,
            "cache_base_dir": expected_training_cache_root,
            "cache_source_stage": "exit",
        },
        {
            "x0": 14.0,
            "horizon": 1,
            "output_dir": str(Path("runs") / "pytest_artifacts" / "sot_strategy_retrain" / "exit" / "exit_start_0004"),
            "n_paths": 3,
            "cache_base_dir": expected_training_cache_root,
            "cache_source_stage": "exit",
        },
    ]

    assert len(strategy.entry_stage_history) == 2
    assert len(strategy.exit_stage_history) == 2
    assert result.step_records[1].decision_metadata["policy_id"] == "sot_entry_0000"
    assert result.step_records[2].decision_metadata["policy_id"] == "sot_exit_0001"


def test_sot_strategy_reuses_cached_stage_features_between_runs(monkeypatch) -> None:
    feature_build_calls: list[dict[str, object]] = []
    entry_training_calls: list[dict[str, object]] = []
    exit_training_calls: list[dict[str, object]] = []

    def fake_sample_stage_paths(config, ou_params, *, x0, horizon):
        paths = np.tile(
            np.linspace(float(x0), float(x0) + float(max(horizon, 0)), int(horizon) + 1, dtype = np.float64),
            (int(config["generator"]["sample"]["n_paths"]), 1),
        )
        request = SimpleNamespace(horizon = int(horizon), x0 = float(x0))
        return paths, request

    def fake_build_real_stage_feature_tensor(spread_segment, formation_spread, config):
        segment = np.asarray(spread_segment, dtype = np.float64)
        feature_build_calls.append(
            {
                "segment_length": int(len(segment)),
                "x0": float(segment[0]),
            }
        )
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
        entry_training_calls.append(
            {
                "cache_base_dir": None if cache_base_dir is None else str(cache_base_dir),
                "cache_source_stage": None if cache_source is None else cache_source.get("stage"),
                "cache_source_kind": None if cache_source is None else cache_source.get("cache_kind"),
                "extra_episode_stage": None
                if extra_metadata is None
                else extra_metadata.get("episode", {}).get("stage"),
            }
        )
        return SimpleNamespace(
            best_policy = LinearStoppingPolicy(weights = np.array([1.0], dtype = np.float32)),
            artifacts = SimpleNamespace(output_dir = output_dir),
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
        exit_training_calls.append(
            {
                "cache_base_dir": None if cache_base_dir is None else str(cache_base_dir),
                "cache_source_stage": None if cache_source is None else cache_source.get("stage"),
                "cache_source_kind": None if cache_source is None else cache_source.get("cache_kind"),
                "extra_episode_stage": None
                if extra_metadata is None
                else extra_metadata.get("episode", {}).get("stage"),
            }
        )
        return SimpleNamespace(
            best_policy = LinearStoppingPolicy(weights = np.array([1.0], dtype = np.float32)),
            artifacts = SimpleNamespace(output_dir = output_dir),
        )

    monkeypatch.setattr("src.sigstop.backtest.sot.sample_sot_stage_paths", fake_sample_stage_paths)
    monkeypatch.setattr("src.sigstop.backtest.sot.build_real_stage_feature_tensor", fake_build_real_stage_feature_tensor)
    monkeypatch.setattr("src.sigstop.backtest.sot.train_entry_policy", fake_train_entry_policy)
    monkeypatch.setattr("src.sigstop.backtest.sot.train_exit_policy", fake_train_exit_policy)

    config = _toy_config()
    cache_root = Path("runs") / "pytest_artifacts" / f"sot_stage_feature_cache_{uuid.uuid4().hex}"
    config["features"]["artifacts_dir"] = str(cache_root)
    trading_window = _toy_window(n_days = 3)
    formation_spread = np.array([9.0, 10.0, 11.0, 12.0], dtype = np.float64)
    ou_params = _toy_ou_params()

    first_strategy = SOTBacktestStrategy(
        config = config,
        formation_spread = formation_spread,
        ou_params = ou_params,
        output_root = Path("runs") / "pytest_artifacts" / "first" / "sot",
        run_id = "pytest_sot_cache_first",
    )
    first_result = run_backtest_engine(
        trading_window,
        first_strategy,
        engine_config = BacktestEngineConfig(trading_days = 3),
    )

    assert [record.action for record in first_result.step_records] == [
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
        StrategyAction.EXIT_LONG_SPREAD,
    ]
    assert len(feature_build_calls) == 2
    assert first_strategy.entry_stage_history[0].feature_cache_hit is False
    assert first_strategy.exit_stage_history[0].feature_cache_hit is False

    second_strategy = SOTBacktestStrategy(
        config = config,
        formation_spread = formation_spread,
        ou_params = ou_params,
        output_root = Path("runs") / "pytest_artifacts" / "second" / "sot",
        run_id = "pytest_sot_cache_second",
    )
    second_result = run_backtest_engine(
        trading_window,
        second_strategy,
        engine_config = BacktestEngineConfig(trading_days = 3),
    )

    assert [record.action for record in second_result.step_records] == [
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
        StrategyAction.EXIT_LONG_SPREAD,
    ]
    assert len(feature_build_calls) == 2
    assert second_strategy.entry_stage_history[0].feature_cache_hit is True
    assert second_strategy.exit_stage_history[0].feature_cache_hit is True
    assert second_strategy.entry_stage_history[0].feature_cache_path is not None
    assert second_strategy.exit_stage_history[0].feature_cache_path is not None
    assert second_strategy.entry_stage_history[0].feature_cache_path.exists()
    assert second_strategy.exit_stage_history[0].feature_cache_path.exists()
    expected_training_cache_root = str((Path.cwd() / "data" / "synthetic" / "gs_ms" / "ou").resolve())
    assert entry_training_calls == [
        {
            "cache_base_dir": expected_training_cache_root,
            "cache_source_stage": "entry",
            "cache_source_kind": "sot_runtime_ou_sample",
            "extra_episode_stage": "entry",
        },
        {
            "cache_base_dir": expected_training_cache_root,
            "cache_source_stage": "entry",
            "cache_source_kind": "sot_runtime_ou_sample",
            "extra_episode_stage": "entry",
        },
    ]
    assert exit_training_calls == [
        {
            "cache_base_dir": expected_training_cache_root,
            "cache_source_stage": "exit",
            "cache_source_kind": "sot_runtime_ou_sample",
            "extra_episode_stage": "exit",
        },
        {
            "cache_base_dir": expected_training_cache_root,
            "cache_source_stage": "exit",
            "cache_source_kind": "sot_runtime_ou_sample",
            "extra_episode_stage": "exit",
        },
    ]
