from __future__ import annotations
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.sigstop.config import load_config
from src.sigstop.features.feature_builder import build_batched_feature_tensor
from src.sigstop.features.manifest import compute_file_sha256
from src.sigstop.features.scaling import fit_zscore_scaler, scaler_to_dict
from src.sigstop.generators.cache import build_ou_sample_cache_paths, load_sample_cache
from src.sigstop.generators.ou_sim import resolve_x0_from_spread_artifact
from src.sigstop.generators.spec import build_default_sample_request, build_generator_context
from src.sigstop.paths import ensure_directories
from src.sigstop.stopping.payoff import build_entry_payoff_process
from src.sigstop.train.synthetic_cache import load_or_build_synthetic_training_data
from src.sigstop.train.trainer import StoppingTrainingData, TrainingResult, build_default_stage_output_dir, build_training_config_from_dict, train_linear_stopping_policy


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Build entry-stage features and payoff paths from simulated spreads
def build_entry_training_data(
    spread_paths: np.ndarray,
    formation_spread: np.ndarray,
    config: dict[str, Any],
) -> StoppingTrainingData:
    scaler = fit_zscore_scaler(formation_spread)

    _raw_path_batch_size = _get_config_value(config, ["features", "signature", "path_batch_size"], None)
    path_batch_size = int(_raw_path_batch_size) if _raw_path_batch_size is not None else None

    feature_result = build_batched_feature_tensor(
        spread_paths,
        scaler,
        depth = int(_get_config_value(config, ["features", "signature", "depth"], 4)),
        min_prefix = int(_get_config_value(config, ["features", "signature", "min_prefix"], 2)),
        scalar_term = bool(
            _get_config_value(config, ["features", "signature", "scalar_term"], True)
        ),
        dtype = str(_get_config_value(config, ["features", "signature", "dtype"], "float32")),
        time_representation = str(
            _get_config_value(
                config,
                ["features", "augmentation", "time_representation"],
                "index_0_1",
            )
        ),
        library = str(_get_config_value(config, ["features", "signature", "library"], "esig")),
        device = str(_get_config_value(config, ["features", "signature", "device"], "cpu")),
        basepoint = bool(_get_config_value(config, ["features", "signature", "basepoint"], False)),
        mode = "prefix",
        path_batch_size = path_batch_size,
    )

    dt = float(_get_config_value(config, ["generator", "fit", "dt"], 1.0))
    payoffs = build_entry_payoff_process(
        spread_paths,
        r = float(_get_config_value(config, ["stopping", "payoff", "entry", "r"], 0.0)),
        c = float(_get_config_value(config, ["stopping", "payoff", "entry", "c"], 0.0)),
        dt = dt,
    )

    return StoppingTrainingData(
        features = feature_result.features,
        payoffs = payoffs,
        prefix_ends = feature_result.prefix_ends,
        feature_spec = feature_result.feature_spec,
        scaler_spec = scaler_to_dict(scaler),
        metadata = {
            "stage": "entry",
            "payoff_formula": "exp(-r t) * (-X - c)",
            "paper_params": {
                "k": float(_get_config_value(config, ["stopping", "deterministic_threshold", "k"], 0.05)),
                "mu": float(_get_config_value(config, ["stopping", "deterministic_threshold", "mu"], 20.0)),
            },
        },
    )


# Load the default entry training dataset from cached OU samples
def load_default_entry_training_data(config: dict[str, Any]) -> StoppingTrainingData:
    context = build_generator_context(config)
    x0 = resolve_x0_from_spread_artifact(
        context.trading_spread_path,
        spread_column = "spread",
        row_index = 0,
    )
    sample_request = build_default_sample_request(x0 = x0, config = config, seed = context.run_seed)
    pair_name = "_".join(symbol.upper() for symbol in context.pair)
    sample_path, sample_manifest_path = build_ou_sample_cache_paths(
        context.synthetic_dir,
        pair_name = pair_name,
        n_paths = sample_request.n_paths,
        horizon = sample_request.horizon,
        dt = sample_request.dt,
        seed = sample_request.seed,
    )
    source_metadata = {
        "sample_path": str(sample_path),
        "sample_hash": compute_file_sha256(sample_path),
        "sample_manifest_path": str(sample_manifest_path),
        "formation_spread_path": str(context.formation_spread_path),
        "formation_spread_hash": compute_file_sha256(context.formation_spread_path),
    }

    training_data, _, _, _ = load_or_build_synthetic_training_data(
        stage = "entry",
        config = config,
        cache_base_dir = context.synthetic_dir,
        source = source_metadata,
        builder = lambda: build_entry_training_data(
            load_sample_cache(sample_path)["paths"],
            pd.read_parquet(context.formation_spread_path)["spread"].to_numpy(),
            config,
        ),
        extra_metadata = {"source": source_metadata},
    )
    return training_data


# Train the entry stopping policy using either provided or cached data
def train_entry_policy(
    config_path: str | Path | None = None,
    *,
    config: dict[str, Any] | None = None,
    output_dir: str | Path | None = None,
    run_id: str | None = None,
    spread_paths: np.ndarray | None = None,
    formation_spread: np.ndarray | None = None,
    resume_from: str | Path | None = None,
    cache_base_dir: str | Path | None = None,
    cache_source: dict[str, Any] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> TrainingResult:
    if config is not None and config_path is not None:
        raise ValueError("Provide either config or config_path, not both.")

    resolved_config = (
        dict(config)
        if config is not None
        else (load_config(config_path) if config_path is not None else load_config())
    )
    ensure_directories()

    if spread_paths is not None:
        if formation_spread is None:
            raise ValueError("formation_spread is required when spread_paths are provided.")
        if cache_base_dir is not None and cache_source is not None:
            training_data, _, _, _ = load_or_build_synthetic_training_data(
                stage = "entry",
                config = resolved_config,
                cache_base_dir = cache_base_dir,
                source = cache_source,
                builder = lambda: build_entry_training_data(
                    spread_paths,
                    formation_spread,
                    resolved_config,
                ),
                extra_metadata = extra_metadata,
            )
        else:
            training_data = build_entry_training_data(
                spread_paths,
                formation_spread,
                resolved_config,
            )
            if extra_metadata is not None:
                training_data.metadata.update(extra_metadata)
    else:
        training_data = load_default_entry_training_data(resolved_config)

    training_config = build_training_config_from_dict(resolved_config)
    resolved_run_id = run_id
    resolved_output_dir = None
    if output_dir is not None:
        resolved_output_dir = Path(output_dir)
    else:
        resolved_run_id, resolved_output_dir = build_default_stage_output_dir(
            "entry",
            run_id = run_id,
        )

    training_data.metadata["feature_spec"] = training_data.feature_spec
    training_data.metadata["scaler_spec"] = training_data.scaler_spec

    return train_linear_stopping_policy(
        training_data.features,
        training_data.payoffs,
        training_config,
        output_dir = resolved_output_dir,
        stage = "entry",
        run_id = resolved_run_id,
        resume_from = resume_from,
        extra_manifest_data = training_data.metadata,
    )
