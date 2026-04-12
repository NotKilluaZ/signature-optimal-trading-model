from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from src.sigstop.features.augmentation import augment_batched_spreads_with_time, augment_spread_with_time
from src.sigstop.features.scaling import ZScoreScaler, apply_batched_zscore_scaler, apply_zscore_scaler
from src.sigstop.features.signature_extractor import compute_prefix_signatures, expected_signature_dim


@dataclass
class FeatureBuildResult:
    features: np.ndarray
    scaled_spread: np.ndarray
    augmented_path: np.ndarray
    prefix_ends: np.ndarray
    feature_spec: dict[str, Any]


@dataclass
class BatchedFeatureBuildResult:
    features: np.ndarray
    scaled_spreads: np.ndarray
    augmented_paths: np.ndarray
    prefix_ends: np.ndarray
    feature_spec: dict[str, Any]

# Validate feature output
def _validate_feature_output(
    features: np.ndarray,
    *,
    channels: int,
    depth: int,
    scalar_term: bool,
) -> None:
    expected_dim = expected_signature_dim(
        channels = channels,
        depth = depth,
        scalar_term = scalar_term,
    )

    if features.shape[-1] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch. Expected: {expected_dim}, "
            f"Got: {features.shape[-1]}"
        )
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature tensor contains NaN or infinite value(s)")

# Build feature specifications in dictionary
def _build_feature_spec(
    *,
    augmented_shape: tuple[int, ...],
    features_shape: tuple[int, ...],
    depth: int,
    scalar_term: bool,
    min_prefix: int,
    time_representation: str,
    mode: str,
    signature_spec: dict[str, Any],
) -> dict[str, Any]:
    channels = int(augmented_shape[-1])

    feature_spec = {
        "augmentation": {
            "add_time": True,
            "time_representation": time_representation
        },
        "channels_C": channels,
        "depth_N": int(depth),
        "scalar_term_included": bool(scalar_term),
        "prefix_mode": mode == "prefix",
        "min_prefix": int(min_prefix),
        "prefix_ends": signature_spec["prefix_ends"],
        "dtype": str(np.dtype(signature_spec["dtype"]).name),
        "input_shape": list(augmented_shape),
        "output_shape": list(features_shape),
        "library": signature_spec["library"],
        "library_version": signature_spec["library_version"],
        "device": signature_spec["device"],
        "basepoint": signature_spec["basepoint"],
        "mode": mode
    }

    if len(augmented_shape) == 3:
        feature_spec["batch_size_B"] = int(augmented_shape[0])
        feature_spec["path_length_L"] = int(augmented_shape[1])

    return feature_spec


# Combine all other feature files/functions into one main pipeline
# Pipeline: Input raw spread -> scale with z-score -> augment path -> prefix signatures
def build_feature_tensor(
    spread: pd.Series | np.ndarray,
    scaler: ZScoreScaler,
    depth: int,
    *,
    min_prefix: int = 2,
    prefix_ends: np.ndarray | list[int] | None = None,
    scalar_term: bool = True,
    dtype: str | np.dtype | type[np.floating] = np.float32,
    time_representation: str = "index_0_1",
    library: str = "esig",
    device: str = "cpu",
    mode: str = "prefix",
    basepoint: bool = False
) -> FeatureBuildResult:
    resolved_dtype = np.dtype(dtype)

    # Step 1: Scale spread using formation-fitted statistics
    scaled_spread = apply_zscore_scaler(spread, scaler, resolved_dtype)

    # Step 2: Augment spread with time -> path of shape (L, 2)
    augmented_path = augment_spread_with_time(
        scaled_spread,
        dtype = resolved_dtype,
        time_representation = time_representation
    )

    # Step 3: Extract prefix signatures
    features, signature_spec = compute_prefix_signatures(
        augmented_path = augmented_path,
        depth = depth,
        scalar_term = scalar_term,
        dtype = resolved_dtype,
        device = device,
        basepoint = basepoint,
        min_prefix = min_prefix,
        prefix_ends = prefix_ends,
        mode = mode,
        library = library
    )

    # Step 4: Validate dimension and finiteness
    _validate_feature_output(
        features,
        channels = augmented_path.shape[1],
        depth = depth,
        scalar_term = scalar_term,
    )
    feature_spec = _build_feature_spec(
        augmented_shape = tuple(augmented_path.shape),
        features_shape = tuple(features.shape),
        depth = depth,
        scalar_term = scalar_term,
        min_prefix = min_prefix,
        time_representation = time_representation,
        mode = mode,
        signature_spec = signature_spec,
    )

    return FeatureBuildResult(
        features = features.astype(resolved_dtype, copy = False),
        scaled_spread = scaled_spread.astype(resolved_dtype, copy = False),
        augmented_path = augmented_path.astype(resolved_dtype, copy = False),
        prefix_ends = np.asarray(signature_spec["prefix_ends"], dtype = np.int32),
        feature_spec = feature_spec
    )

# Build batched feature tensor
def build_batched_feature_tensor(
    spreads: np.ndarray | list[list[float]],
    scaler: ZScoreScaler,
    depth: int,
    *,
    min_prefix: int = 2,
    prefix_ends: np.ndarray | list[int] | None = None,
    scalar_term: bool = True,
    dtype: str | np.dtype | type[np.floating] = np.float32,
    time_representation: str = "index_0_1",
    library: str = "esig",
    device: str = "cpu",
    mode: str = "prefix",
    basepoint: bool = False,
    path_batch_size: int | None = None,
) -> BatchedFeatureBuildResult:
    resolved_dtype = np.dtype(dtype)

    # If path_batch_size is set and the batch is large, process in chunks to
    # avoid allocating the full (N, P, D) feature tensor in one shot
    if path_batch_size is not None:
        spreads_arr = np.asarray(spreads)
        n_paths = spreads_arr.shape[0]
        if n_paths > path_batch_size:
            chunks_features: list[np.ndarray] = []
            chunks_scaled: list[np.ndarray] = []
            chunks_augmented: list[np.ndarray] = []
            first_spec: dict[str, Any] | None = None
            shared_prefix_ends: np.ndarray | None = None

            for start in range(0, n_paths, path_batch_size):
                chunk = spreads_arr[start : start + path_batch_size]
                chunk_result = build_batched_feature_tensor(
                    chunk, scaler, depth,
                    min_prefix = min_prefix,
                    prefix_ends = prefix_ends,
                    scalar_term = scalar_term,
                    dtype = resolved_dtype,
                    time_representation = time_representation,
                    library = library,
                    device = device,
                    mode = mode,
                    basepoint = basepoint,
                    path_batch_size = None,
                )
                chunks_features.append(chunk_result.features)
                chunks_scaled.append(chunk_result.scaled_spreads)
                chunks_augmented.append(chunk_result.augmented_paths)
                if first_spec is None:
                    first_spec = chunk_result.feature_spec
                    shared_prefix_ends = chunk_result.prefix_ends

            features = np.concatenate(chunks_features, axis = 0)
            scaled_spreads_out = np.concatenate(chunks_scaled, axis = 0)
            augmented_paths_out = np.concatenate(chunks_augmented, axis = 0)

            full_spec = dict(first_spec)  # type: ignore[arg-type]
            full_spec["batch_size_B"] = n_paths
            full_spec["input_shape"] = [n_paths] + list(first_spec["input_shape"][1:])  # type: ignore[index]
            full_spec["output_shape"] = list(features.shape)

            return BatchedFeatureBuildResult(
                features = features,
                scaled_spreads = scaled_spreads_out,
                augmented_paths = augmented_paths_out,
                prefix_ends = shared_prefix_ends,  # type: ignore[arg-type]
                feature_spec = full_spec,
            )

    # Step 1: Scale each simulated spread path with the same formation-fitted scaler
    scaled_spreads = apply_batched_zscore_scaler(spreads, scaler, resolved_dtype)

    # Step 2: Add the shared deterministic time channel to each path
    augmented_paths = augment_batched_spreads_with_time(
        scaled_spreads,
        dtype = resolved_dtype,
        time_representation = time_representation,
    )

    # Step 3: Extract prefix signatures for each path in the batch
    features, signature_spec = compute_prefix_signatures(
        augmented_path = augmented_paths,
        depth = depth,
        scalar_term = scalar_term,
        dtype = resolved_dtype,
        device = device,
        basepoint = basepoint,
        min_prefix = min_prefix,
        prefix_ends = prefix_ends,
        mode = mode,
        library = library,
    )

    _validate_feature_output(
        features,
        channels = augmented_paths.shape[2],
        depth = depth,
        scalar_term = scalar_term,
    )
    feature_spec = _build_feature_spec(
        augmented_shape = tuple(augmented_paths.shape),
        features_shape = tuple(features.shape),
        depth = depth,
        scalar_term = scalar_term,
        min_prefix = min_prefix,
        time_representation = time_representation,
        mode = mode,
        signature_spec = signature_spec,
    )

    return BatchedFeatureBuildResult(
        features = features.astype(resolved_dtype, copy = False),
        scaled_spreads = scaled_spreads.astype(resolved_dtype, copy = False),
        augmented_paths = augmented_paths.astype(resolved_dtype, copy = False),
        prefix_ends = np.asarray(signature_spec["prefix_ends"], dtype = np.int32),
        feature_spec = feature_spec,
    )
