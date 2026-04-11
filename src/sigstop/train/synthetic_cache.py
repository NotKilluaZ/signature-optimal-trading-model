from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
import numpy as np
from src.sigstop.features.manifest import summarize_array
from src.sigstop.train.trainer import StoppingTrainingData


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Normalize cache metadata into JSON-serializable values
def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_json_value(item)
            for key, item in sorted(value.items(), key = lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


# Build the deterministic cache inputs for one synthetic training dataset
def build_synthetic_training_cache_inputs(
    *,
    stage: str,
    config: dict[str, Any],
    source: dict[str, Any],
) -> dict[str, Any]:
    if stage not in {"entry", "exit"}:
        raise ValueError(f"Unsupported synthetic training cache stage: {stage!r}")

    payoff_config = (
        {
            "r": float(_get_config_value(config, ["stopping", "payoff", "entry", "r"], 0.0)),
            "c": float(_get_config_value(config, ["stopping", "payoff", "entry", "c"], 0.0)),
        }
        if stage == "entry"
        else {
            "r_hat": float(_get_config_value(config, ["stopping", "payoff", "exit", "r_hat"], 0.0)),
            "c_hat": float(_get_config_value(config, ["stopping", "payoff", "exit", "c_hat"], 0.0)),
        }
    )

    return {
        "stage": stage,
        "feature_config": {
            "depth": int(_get_config_value(config, ["features", "signature", "depth"], 4)),
            "min_prefix": int(_get_config_value(config, ["features", "signature", "min_prefix"], 2)),
            "scalar_term": bool(_get_config_value(config, ["features", "signature", "scalar_term"], True)),
            "dtype": str(_get_config_value(config, ["features", "signature", "dtype"], "float32")),
            "library": str(_get_config_value(config, ["features", "signature", "library"], "esig")),
            "device": str(_get_config_value(config, ["features", "signature", "device"], "cpu")),
            "basepoint": bool(_get_config_value(config, ["features", "signature", "basepoint"], False)),
            "time_representation": str(
                _get_config_value(
                    config,
                    ["features", "augmentation", "time_representation"],
                    "index_0_1",
                )
            ),
            "spread_scaler": str(
                _get_config_value(config, ["features", "scaling", "spread_scaler"], "zscore")
            ),
            "scaling_fit_on": str(
                _get_config_value(config, ["features", "scaling", "fit_on"], "formation_only")
            ),
            "mode": "prefix",
        },
        "payoff_config": {
            **payoff_config,
            "dt": float(_get_config_value(config, ["generator", "fit", "dt"], 1.0)),
        },
        "source": _normalize_json_value(source),
    }


# Hash normalized cache inputs into a stable cache key
def build_synthetic_training_cache_key(cache_inputs: dict[str, Any]) -> str:
    serialized = json.dumps(
        _normalize_json_value(cache_inputs),
        sort_keys = True,
        separators = (",", ":"),
        ensure_ascii = True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


# Build deterministic paths for one synthetic training cache artifact pair
def build_synthetic_training_cache_paths(
    base_dir: str | Path,
    *,
    stage: str,
    cache_key: str,
) -> tuple[Path, Path]:
    root = Path(base_dir) / "stopping_policy_cache" / stage
    data_path = root / f"{stage}_training_{cache_key}.npz"
    manifest_path = root / f"manifest_{stage}_training_{cache_key}.json"
    return data_path, manifest_path


# Save cached synthetic training arrays and metadata to disk
def save_synthetic_training_cache(
    data_path: str | Path,
    manifest_path: str | Path,
    *,
    cache_key: str,
    cache_inputs: dict[str, Any],
    training_data: StoppingTrainingData,
) -> None:
    resolved_data_path = Path(data_path)
    resolved_manifest_path = Path(manifest_path)
    resolved_data_path.parent.mkdir(parents = True, exist_ok = True)
    resolved_manifest_path.parent.mkdir(parents = True, exist_ok = True)

    features = np.asarray(training_data.features)
    payoffs = np.asarray(training_data.payoffs)
    prefix_ends = np.asarray(training_data.prefix_ends, dtype = np.int32)

    if not np.all(np.isfinite(features)):
        raise ValueError("Synthetic training features contain NaN or infinite value(s).")
    if not np.all(np.isfinite(payoffs)):
        raise ValueError("Synthetic training payoffs contain NaN or infinite value(s).")

    np.savez(
        resolved_data_path,
        features = features,
        payoffs = payoffs,
        prefix_ends = prefix_ends,
    )

    manifest = {
        "stage": str(cache_inputs["stage"]),
        "cache_key": str(cache_key),
        "cache_inputs": _normalize_json_value(cache_inputs),
        "feature_spec": _normalize_json_value(training_data.feature_spec),
        "scaler_spec": _normalize_json_value(training_data.scaler_spec),
        "metadata": _normalize_json_value(training_data.metadata),
        "array_facts": {
            "features": summarize_array(features),
            "payoffs": summarize_array(payoffs),
            "prefix_ends": {
                "dtype": str(prefix_ends.dtype),
                "shape": list(prefix_ends.shape),
                "min": int(np.min(prefix_ends)) if prefix_ends.size else None,
                "max": int(np.max(prefix_ends)) if prefix_ends.size else None,
            },
        },
    }
    with open(resolved_manifest_path, "w", encoding = "utf-8") as handle:
        json.dump(manifest, handle, indent = 2)


# Load a cached synthetic training dataset from disk
def load_synthetic_training_cache(
    data_path: str | Path,
    manifest_path: str | Path,
) -> StoppingTrainingData:
    resolved_data_path = Path(data_path)
    resolved_manifest_path = Path(manifest_path)

    if not resolved_data_path.exists():
        raise FileNotFoundError(f"Synthetic training cache not found: {resolved_data_path}")
    if not resolved_manifest_path.exists():
        raise FileNotFoundError(
            f"Synthetic training cache manifest not found: {resolved_manifest_path}"
        )

    with np.load(resolved_data_path, allow_pickle = False) as data:
        required_keys = {"features", "payoffs", "prefix_ends"}
        missing = required_keys.difference(data.files)
        if missing:
            raise ValueError(
                "Synthetic training cache is missing required arrays: "
                f"{sorted(missing)}"
            )
        arrays = {key: data[key] for key in data.files}

    with open(resolved_manifest_path, "r", encoding = "utf-8") as handle:
        manifest = json.load(handle)

    return StoppingTrainingData(
        features = np.asarray(arrays["features"]),
        payoffs = np.asarray(arrays["payoffs"]),
        prefix_ends = np.asarray(arrays["prefix_ends"], dtype = np.int32),
        feature_spec = dict(manifest["feature_spec"]),
        scaler_spec = dict(manifest["scaler_spec"]),
        metadata = dict(manifest.get("metadata", {})),
    )


# Reuse a cached synthetic training dataset or build and save it once
def load_or_build_synthetic_training_data(
    *,
    stage: str,
    config: dict[str, Any],
    cache_base_dir: str | Path,
    source: dict[str, Any],
    builder: Callable[[], StoppingTrainingData],
    extra_metadata: dict[str, Any] | None = None,
) -> tuple[StoppingTrainingData, Path, Path, bool]:
    cache_inputs = build_synthetic_training_cache_inputs(
        stage = stage,
        config = config,
        source = source,
    )
    cache_key = build_synthetic_training_cache_key(cache_inputs)
    data_path, manifest_path = build_synthetic_training_cache_paths(
        cache_base_dir,
        stage = stage,
        cache_key = cache_key,
    )

    cache_hit = data_path.exists() and manifest_path.exists()
    if cache_hit:
        training_data = load_synthetic_training_cache(data_path, manifest_path)
    else:
        training_data = builder()
        if extra_metadata:
            training_data.metadata.update(_normalize_json_value(extra_metadata))
        save_synthetic_training_cache(
            data_path,
            manifest_path,
            cache_key = cache_key,
            cache_inputs = cache_inputs,
            training_data = training_data,
        )

    if extra_metadata:
        training_data.metadata.update(_normalize_json_value(extra_metadata))
    training_data.metadata["cache"] = {
        "enabled": True,
        "hit": bool(cache_hit),
        "cache_key": cache_key,
        "data_path": str(data_path),
        "manifest_path": str(manifest_path),
    }

    return training_data, data_path, manifest_path, cache_hit
