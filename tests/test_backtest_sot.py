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


def _install_fake_pretrained_sot_policies(monkeypatch) -> dict[str, list[dict[str, object]]]:
    dataset_calls: list[dict[str, object]] = []
    training_calls: list[dict[str, object]] = []

    def _fake_training_data(stage: str, spread_paths, formation_spread, config):
        paths = np.asarray(spread_paths, dtype = np.float64)
        dataset_calls.append(
            {
                "stage": stage,
                "n_paths": int(paths.shape[0]),
                "horizon": int(paths.shape[1] - 1),
            }
        )
        return SimpleNamespace(
            features = np.ones((int(paths.shape[0]), 1, 1), dtype = np.float32),
            payoffs = np.ones((int(paths.shape[0]), 2), dtype = np.float32),
            prefix_ends = np.array([1], dtype = np.int32),
            feature_spec = {"stage": stage, "feature_dim": 1},
            scaler_spec = {"stage": stage},
            metadata = {"stage": stage},
        )

    def fake_build_entry_training_data(spread_paths, formation_spread, config):
        return _fake_training_data("entry", spread_paths, formation_spread, config)

    def fake_build_exit_training_data(spread_paths, formation_spread, config):
        return _fake_training_data("exit", spread_paths, formation_spread, config)

    def fake_train_linear_stopping_policy(
        features,
        payoffs,
        config,
        *,
        output_dir = None,
        stage = "entry",
        run_id = None,
        initial_policy = None,
        resume_from = None,
        extra_manifest_data = None,
    ):
        feature_dim = int(np.asarray(features).shape[-1])
        output_dir = Path(output_dir)
        output_dir.mkdir(parents = True, exist_ok = True)
        manifest_path = output_dir / "manifest.json"
        policy_best_path = output_dir / "policy_best.pt"
        manifest_path.write_text(f'{{"stage":"{stage}"}}', encoding = "utf-8")
        policy_best_path.write_text(f"{stage}-policy", encoding = "utf-8")
        training_calls.append(
            {
                "stage": stage,
                "feature_dim": feature_dim,
                "output_dir": str(output_dir),
            }
        )
        return SimpleNamespace(
            best_policy = LinearStoppingPolicy(weights = np.ones(feature_dim, dtype = np.float32)),
            artifacts = SimpleNamespace(
                output_dir = output_dir,
                manifest_path = manifest_path,
                policy_best_path = policy_best_path,
            ),
        )

    monkeypatch.setattr("src.sigstop.backtest.sot.build_entry_training_data", fake_build_entry_training_data)
    monkeypatch.setattr("src.sigstop.backtest.sot.build_exit_training_data", fake_build_exit_training_data)
    monkeypatch.setattr("src.sigstop.backtest.sot.train_linear_stopping_policy", fake_train_linear_stopping_policy)
    return {
        "dataset_calls": dataset_calls,
        "training_calls": training_calls,
    }


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


def test_sot_strategy_pretrains_once_per_strategy_and_reuses_inside_engine(monkeypatch) -> None:
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

    pretrain_calls = _install_fake_pretrained_sot_policies(monkeypatch)
    monkeypatch.setattr("src.sigstop.backtest.sot.build_real_stage_feature_tensor", fake_build_real_stage_feature_tensor)

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
        StrategyAction.HOLD,
        StrategyAction.EXIT_LONG_SPREAD,
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
    ]

    assert pretrain_calls["dataset_calls"] == [
        {
            "stage": "entry",
            "n_paths": 3,
            "horizon": 252,
        },
        {
            "stage": "exit",
            "n_paths": 3,
            "horizon": 252,
        },
    ]
    assert pretrain_calls["training_calls"] == [
        {
            "stage": "entry",
            "feature_dim": 1,
            "output_dir": str(Path("runs") / "pytest_artifacts" / "sot_strategy_retrain" / "pretrain" / "entry"),
        },
        {
            "stage": "exit",
            "feature_dim": 1,
            "output_dir": str(Path("runs") / "pytest_artifacts" / "sot_strategy_retrain" / "pretrain" / "exit"),
        },
    ]

    assert len(strategy.entry_stage_history) == 2
    assert len(strategy.exit_stage_history) == 1
    assert strategy.entry_stage_history[0].training_result is strategy.pretrained_entry_result
    assert strategy.exit_stage_history[0].training_result is strategy.pretrained_exit_result
    assert strategy.pretrained_entry_result.artifacts.manifest_path.exists()
    assert strategy.pretrained_exit_result.artifacts.manifest_path.exists()
    assert result.step_records[1].decision_metadata["policy_id"] == "sot_entry_0000"
    assert result.step_records[3].decision_metadata["policy_id"] == "sot_exit_0002"


def test_sot_strategy_reuses_cached_stage_features_between_runs(monkeypatch) -> None:
    feature_build_calls: list[dict[str, object]] = []

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

    pretrain_calls = _install_fake_pretrained_sot_policies(monkeypatch)
    monkeypatch.setattr("src.sigstop.backtest.sot.build_real_stage_feature_tensor", fake_build_real_stage_feature_tensor)

    config = _toy_config()
    cache_root = Path("runs") / "pytest_artifacts" / f"sot_stage_feature_cache_{uuid.uuid4().hex}"
    config["features"]["artifacts_dir"] = str(cache_root)
    trading_window = _toy_window(n_days = 4)
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
        engine_config = BacktestEngineConfig(trading_days = 4),
    )

    assert [record.action for record in first_result.step_records] == [
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
        StrategyAction.HOLD,
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
        engine_config = BacktestEngineConfig(trading_days = 4),
    )

    assert [record.action for record in second_result.step_records] == [
        StrategyAction.HOLD,
        StrategyAction.ENTER_LONG_SPREAD,
        StrategyAction.HOLD,
        StrategyAction.EXIT_LONG_SPREAD,
    ]
    assert len(feature_build_calls) == 2
    assert second_strategy.entry_stage_history[0].feature_cache_hit is True
    assert second_strategy.exit_stage_history[0].feature_cache_hit is True
    assert second_strategy.entry_stage_history[0].feature_cache_path is not None
    assert second_strategy.exit_stage_history[0].feature_cache_path is not None
    assert second_strategy.entry_stage_history[0].feature_cache_path.exists()
    assert second_strategy.exit_stage_history[0].feature_cache_path.exists()
    assert second_strategy.entry_stage_history[0].feature_manifest_path is not None
    assert second_strategy.exit_stage_history[0].feature_manifest_path is not None
    assert second_strategy.entry_stage_history[0].feature_manifest_path.exists()
    assert second_strategy.exit_stage_history[0].feature_manifest_path.exists()
    assert pretrain_calls["dataset_calls"] == [
        {
            "stage": "entry",
            "n_paths": 3,
            "horizon": 252,
        },
        {
            "stage": "exit",
            "n_paths": 3,
            "horizon": 252,
        },
        {
            "stage": "entry",
            "n_paths": 3,
            "horizon": 252,
        },
        {
            "stage": "exit",
            "n_paths": 3,
            "horizon": 252,
        },
    ]
