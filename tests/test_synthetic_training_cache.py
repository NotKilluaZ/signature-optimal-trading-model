from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np

import src.sigstop.train.train_exit as train_exit_module
from src.sigstop.stopping.policy import LinearStoppingPolicy
from src.sigstop.train.synthetic_cache import load_or_build_synthetic_training_data
from src.sigstop.train.trainer import (
    StoppingTrainingData,
    TrainingHistory,
    TrainingResult,
)


def _make_test_dir() -> Path:
    path = Path("runs") / "pytest_artifacts" / f"synthetic_cache_{uuid4().hex}"
    path.mkdir(parents = True, exist_ok = True)
    return path


def _toy_config() -> dict:
    return {
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
            "scaling": {
                "spread_scaler": "zscore",
                "fit_on": "formation_only",
            },
        },
        "generator": {
            "fit": {
                "dt": 1.0,
            }
        },
        "stopping": {
            "deterministic_threshold": {
                "k": 0.05,
                "mu": 20.0,
            },
            "policy": {
                "include_bias": False,
                "scalar_term_in_signatures": True,
                "init": "small_normal",
                "init_weight_scale": 1.0e-2,
            },
            "payoff": {
                "entry": {
                    "r": 0.0,
                    "c": 0.0,
                },
                "exit": {
                    "r_hat": 0.0,
                    "c_hat": 0.0,
                },
            },
        },
        "training": {
            "batch_size": 4,
            "device": "cpu",
            "validation": {
                "fraction": 0.0,
                "shuffle": False,
            },
            "optimizer": {
                "name": "adam",
                "lr": 1.0e-3,
                "betas": [0.9, 0.999],
                "eps": 1.0e-8,
                "weight_decay": 0.0,
            },
            "schedule": {
                "max_epochs": 1,
                "early_stopping": {
                    "enabled": False,
                    "patience": 1,
                    "min_delta": 1.0e-4,
                },
            },
            "regularization": {
                "grad_clip": {
                    "enabled": False,
                    "max_norm": 5.0,
                },
            },
        },
        "repro": {
            "seed": 42,
            "torch_manual_seed": False,
            "deterministic_algorithms": False,
        },
    }


def _dummy_training_data(stage: str) -> StoppingTrainingData:
    return StoppingTrainingData(
        features = np.array(
            [
                [[0.1, 0.2], [0.3, 0.4]],
                [[0.5, 0.6], [0.7, 0.8]],
            ],
            dtype = np.float32,
        ),
        payoffs = np.array(
            [
                [1.0, 1.1, 1.2],
                [1.3, 1.4, 1.5],
            ],
            dtype = np.float64,
        ),
        prefix_ends = np.array([1, 2], dtype = np.int32),
        feature_spec = {
            "depth_N": 2,
            "output_shape": [2, 2, 2],
        },
        scaler_spec = {
            "name": "zscore",
            "mean": 0.0,
            "std": 1.0,
        },
        metadata = {
            "stage": stage,
        },
    )


def test_synthetic_training_cache_round_trips_and_skips_builder_on_hit() -> None:
    config = _toy_config()
    cache_dir = _make_test_dir()
    builder_calls = {"count": 0}
    source = {
        "sample_hash": "abc123",
        "formation_spread_hash": "def456",
    }
    extra_metadata = {
        "source": source,
    }

    def builder() -> StoppingTrainingData:
        builder_calls["count"] += 1
        return _dummy_training_data("entry")

    first, data_path, manifest_path, first_hit = load_or_build_synthetic_training_data(
        stage = "entry",
        config = config,
        cache_base_dir = cache_dir,
        source = source,
        builder = builder,
        extra_metadata = extra_metadata,
    )
    second, _, _, second_hit = load_or_build_synthetic_training_data(
        stage = "entry",
        config = config,
        cache_base_dir = cache_dir,
        source = source,
        builder = builder,
        extra_metadata = extra_metadata,
    )

    assert first_hit is False
    assert second_hit is True
    assert builder_calls["count"] == 1
    assert data_path.exists()
    assert manifest_path.exists()
    np.testing.assert_allclose(first.features, second.features)
    np.testing.assert_allclose(first.payoffs, second.payoffs)
    np.testing.assert_array_equal(first.prefix_ends, second.prefix_ends)
    assert second.metadata["source"]["sample_hash"] == "abc123"
    assert second.metadata["cache"]["hit"] is True


def test_train_exit_policy_reuses_cached_synthetic_training_data(
    monkeypatch,
) -> None:
    config = _toy_config()
    cache_dir = _make_test_dir()
    builder_calls = {"count": 0}
    spread_paths = np.array(
        [
            [10.0, 10.5, 10.25],
            [9.8, 10.1, 10.4],
        ],
        dtype = np.float64,
    )
    formation_spread = np.array([9.5, 10.0, 10.5], dtype = np.float64)
    source = {
        "cache_kind": "runtime_exit_ou_sample",
        "entry_row_index": 12,
        "entry_spread": 10.0,
        "sample_request": {
            "x0": 10.0,
            "horizon": 2,
            "n_paths": 2,
            "seed": 42,
            "dt": 1.0,
            "dtype": "float64",
            "device": "cpu",
            "include_innovations": False,
        },
    }

    def fake_load_config(path = None) -> dict:
        return config

    def fake_build_exit_training_data(
        spread_paths_arg: np.ndarray,
        formation_spread_arg: np.ndarray,
        config_arg: dict,
    ) -> StoppingTrainingData:
        builder_calls["count"] += 1
        np.testing.assert_allclose(spread_paths_arg, spread_paths)
        np.testing.assert_allclose(formation_spread_arg, formation_spread)
        assert config_arg is config
        return _dummy_training_data("exit")

    def fake_train_linear_stopping_policy(
        features: np.ndarray,
        payoffs: np.ndarray,
        training_config,
        **kwargs,
    ) -> TrainingResult:
        policy = LinearStoppingPolicy(weights = np.array([0.0, 0.0], dtype = np.float32))
        return TrainingResult(
            policy = policy,
            best_policy = policy,
            optimizer_state = {},
            history = TrainingHistory(),
            config = training_config,
            artifacts = None,
            train_features_shape = tuple(features.shape),
            train_payoffs_shape = tuple(payoffs.shape),
        )

    monkeypatch.setattr(train_exit_module, "load_config", fake_load_config)
    monkeypatch.setattr(
        train_exit_module,
        "build_exit_training_data",
        fake_build_exit_training_data,
    )
    monkeypatch.setattr(
        train_exit_module,
        "train_linear_stopping_policy",
        fake_train_linear_stopping_policy,
    )

    first = train_exit_module.train_exit_policy(
        spread_paths = spread_paths,
        formation_spread = formation_spread,
        cache_base_dir = cache_dir,
        cache_source = source,
        extra_metadata = {"source": source},
    )
    second = train_exit_module.train_exit_policy(
        spread_paths = spread_paths,
        formation_spread = formation_spread,
        cache_base_dir = cache_dir,
        cache_source = source,
        extra_metadata = {"source": source},
    )

    assert builder_calls["count"] == 1
    assert first.train_features_shape == (2, 2, 2)
    assert second.train_features_shape == (2, 2, 2)
