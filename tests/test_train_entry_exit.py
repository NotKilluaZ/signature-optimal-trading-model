from __future__ import annotations
import numpy as np
from src.sigstop.train.train_entry import build_entry_training_data
from src.sigstop.train.train_exit import build_exit_training_data


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
    }


def test_entry_and_exit_training_builders_produce_aligned_shapes() -> None:
    config = _toy_config()
    formation_spread = np.array([10.0, 11.0, 12.0, 13.0, 14.0], dtype = np.float64)
    spread_paths = np.array(
        [
            [10.0, 10.5, 11.0, 10.0, 9.5],
            [12.0, 11.8, 11.6, 11.7, 11.9],
        ],
        dtype = np.float64,
    )

    entry_data = build_entry_training_data(spread_paths, formation_spread, config)
    exit_data = build_exit_training_data(spread_paths, formation_spread, config)

    assert entry_data.features.shape[:2] == (2, 4)
    assert exit_data.features.shape[:2] == (2, 4)
    assert entry_data.payoffs.shape == (2, 5)
    assert exit_data.payoffs.shape == (2, 5)
    assert entry_data.prefix_ends[0] == 1
    assert exit_data.prefix_ends[-1] == 4
    assert entry_data.metadata["stage"] == "entry"
    assert exit_data.metadata["stage"] == "exit"
