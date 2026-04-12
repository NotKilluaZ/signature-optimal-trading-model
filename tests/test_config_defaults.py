from __future__ import annotations

from src.sigstop.config import load_config


def test_default_config_exposes_backtest_surface() -> None:
    config = load_config()

    assert config["backtest"] == {
        "trading_days": 252,
        "execution_price": "close",
        "allow_forced_exit_end": True,
        "annualization_factor": 252,
    }
    assert config["sot"] == {
        "retrain_cadence": "per_trade",
        "k": 0.05,
        "mu": 20.0,
        "ou_samples_M": 100,
        "feature_depth_N": 4,
        "episode_horizon": 60,
        "pretrain_horizon": 60,
        "cache_episode_features": True,
        "rolling_refit": {
            "enabled": False,
            "window": 60,
        },
    }
    assert config["baseline"] == {
        "ma_window": 100,
        "k": 0.1,
        "min_periods": 100,
        "use_shift_1": True,
    }
    assert config["costs"] == {
        "fixed_per_action": 0.001,
        "proportional": 0.0005,
        "slippage_bps": 0.0,
        "model": "spread_proxy",
    }
    assert config["metrics"] == {
        "rf_rate_daily": 0.0,
        "report": {
            "include_daily": True,
            "include_annualized": True,
        },
    }
    assert config["artifacts"] == {
        "root": "runs",
        "save_trades": True,
        "save_equity_curve": True,
        "save_plots": True,
        "save_manifests": True,
    }


def test_default_backtest_surface_stays_consistent_with_existing_training_defaults() -> None:
    config = load_config()

    assert config["backtest"]["trading_days"] == config["split"]["trading_days"]
    assert config["backtest"]["annualization_factor"] == config["split"]["trading_days"]

    assert config["sot"]["k"] == config["stopping"]["deterministic_threshold"]["k"]
    assert config["sot"]["mu"] == config["stopping"]["deterministic_threshold"]["mu"]
    assert config["sot"]["feature_depth_N"] == config["features"]["signature"]["depth"]
    assert config["sot"]["ou_samples_M"] == 100
    assert config["training"]["M"] == 2000
    assert config["generator"]["sample"]["n_paths"] == 5000
    assert config["sot"]["ou_samples_M"] != config["training"]["M"]
    assert config["sot"]["ou_samples_M"] != config["generator"]["sample"]["n_paths"]
