from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
from src.sigstop.features.feature_builder import FeatureBuildResult
from src.sigstop.features.manifest import load_manifest


# Ensure directories are made if not exisitng
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)


# Normalize cache inputs into JSON-serializable values
def _normalize_json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_json_value(raw_value) for key, raw_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


# Hash one feature-cache input array into a stable digest
def _hash_feature_array(values: np.ndarray) -> str:
    resolved = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(str(resolved.dtype).encode("utf-8"))
    digest.update(np.asarray(resolved.shape, dtype = np.int64).tobytes())
    digest.update(resolved.view(np.uint8).tobytes())
    return digest.hexdigest()


# Save feature arrays to .npz cache file
def save_feature_cache(
    feature_path: str | Path,
    features: np.ndarray,
    prefix_ends: np.ndarray | None = None,
    *,
    scaled_spread: np.ndarray | None = None,
    augmented_path: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
) -> None:
    feature_path = Path(feature_path)
    ensure_parent_dir(feature_path)

    payload: dict[str, np.ndarray] = {
        "features": np.asarray(features),
    }
    if prefix_ends is not None:
        payload["prefix_ends"] = np.asarray(prefix_ends)
    if scaled_spread is not None:
        payload["scaled_spread"] = np.asarray(scaled_spread)
    if augmented_path is not None:
        payload["augmented_path"] = np.asarray(augmented_path)
    if extra_arrays is not None:
        for key, value in extra_arrays.items():
            if value is None:
                continue
            payload[str(key)] = np.asarray(value)

    for key, value in payload.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Feature cache array {key!r} contains NaN or infinite value(s)")

    np.savez(feature_path, **payload)


# Load cached .npz feature array file
def load_feature_cache(feature_path: str | Path) -> dict[str, np.ndarray]:
    feature_path = Path(feature_path)

    if not feature_path.exists():
        raise FileNotFoundError(f"Feature cache not found: {feature_path}")

    with np.load(feature_path, allow_pickle = False) as data:
        return {key: data[key] for key in data.files}


# Save one full feature-build result into the cache format
def save_feature_build_result(feature_path: str | Path, result: FeatureBuildResult) -> None:
    save_feature_cache(
        feature_path,
        result.features,
        prefix_ends = result.prefix_ends,
        scaled_spread = result.scaled_spread,
        augmented_path = result.augmented_path,
    )


# Load one full feature-build result from a cache artifact pair
def load_feature_build_result(
    feature_path: str | Path,
    manifest_path: str | Path,
) -> FeatureBuildResult:
    cache_payload = load_feature_cache(feature_path)
    required_arrays = {"features", "scaled_spread", "augmented_path", "prefix_ends"}
    missing = required_arrays.difference(cache_payload.keys())
    if missing:
        raise ValueError(
            f"Feature cache is missing required arrays: {sorted(missing)}"
        )

    manifest = load_manifest(manifest_path)
    feature_spec = manifest.get("feature_spec")
    if not isinstance(feature_spec, dict):
        raise ValueError("Feature cache manifest is missing a valid 'feature_spec' payload.")

    return FeatureBuildResult(
        features = np.asarray(cache_payload["features"]),
        scaled_spread = np.asarray(cache_payload["scaled_spread"]),
        augmented_path = np.asarray(cache_payload["augmented_path"]),
        prefix_ends = np.asarray(cache_payload["prefix_ends"], dtype = np.int32),
        feature_spec = dict(feature_spec),
    )

# Build deterministic cache inputs for one backtest stage feature tensor
def build_backtest_feature_cache_inputs(
    *,
    namespace: str,
    stage: str,
    state_start_index: int,
    spread_segment: np.ndarray,
    formation_spread: np.ndarray,
    feature_settings: dict[str, Any],
) -> dict[str, Any]:
    return {
        "cache_schema_version": 1,
        "namespace": str(namespace),
        "stage": str(stage),
        "state_start_index": int(state_start_index),
        "spread_segment_length": int(len(spread_segment)),
        "formation_spread_length": int(len(formation_spread)),
        "spread_segment_hash": _hash_feature_array(np.asarray(spread_segment, dtype = np.float64)),
        "formation_spread_hash": _hash_feature_array(np.asarray(formation_spread, dtype = np.float64)),
        "feature_settings": _normalize_json_value(feature_settings),
    }


# Hash normalized backtest feature cache inputs into a stable cache key
def build_backtest_feature_cache_key(cache_inputs: dict[str, Any]) -> str:
    normalized_inputs = _normalize_json_value(cache_inputs)
    payload = json.dumps(normalized_inputs, sort_keys = True, separators = (",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Build deterministic file paths for one backtest stage feature cache artifact pair
def build_backtest_feature_cache_paths(
    base_dir: str | Path,
    *,
    namespace: str,
    stage: str,
    state_start_index: int,
    depth: int,
    cache_key: str,
) -> tuple[Path, Path]:
    base_dir = Path(base_dir)
    stage_dir = base_dir / str(namespace) / str(stage)
    stem = (
        f"{stage}_start_{int(state_start_index):04d}_"
        f"sig_prefix_depth{int(depth)}_{str(cache_key)[:16]}"
    )
    return stage_dir / f"{stem}.npz", stage_dir / f"{stem}.json"
