from __future__ import annotations
from typing import Any
import numpy as np
from src.sigstop.features.signature_extractor import expected_signature_dim

# Raise error if array contains invalid values
def validate_finite_array(arr: np.ndarray, name: str = "array") -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")

# Validate tensor rank
def validate_feature_rank(features: np.ndarray, expected_rank: int | tuple[int, ...]) -> None:
    expected_ranks = ((expected_rank,) if isinstance(expected_rank, int) else tuple(expected_rank))

    if features.ndim not in expected_ranks:
        raise ValueError(
            f"Feature tensor rank mismatch. Got ndim: {features.ndim}, "
            f"expected ndim in: {expected_ranks}."
        )

# Validate final feature dimension matches signature math
def validate_feature_dim_from_spec(features: np.ndarray, channels: int, depth: int, scalar_term: bool) -> None:
    expected_dim = expected_signature_dim(channels = channels, depth = depth, scalar_term = scalar_term)

    actual_dim = features.shape[-1]

    if actual_dim != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch. Got D: {actual_dim}, "
            f"expected D: {expected_dim}."
        )

# Check manifest array data is consistent with referenced feature tensor
def validate_manifest_matches_features(features: np.ndarray, manifest: dict[str, Any]) -> None:
    manifest_shape = manifest["array_facts"]["shape"]
    manifest_dtype = manifest["array_facts"]["dtype"]

    if list(features.shape) != list(manifest_shape):
        raise ValueError(
            f"Manifest shape mismatch. Actual shape: {list(features.shape)}, "
            f"manifest shape: {manifest_shape}."
        )

    if str(features.dtype) != str(manifest_dtype):
        raise ValueError(
            f"Manifest dtype mismatch. Actual dtype: {features.dtype}, "
            f"manifest dtype: {manifest_dtype}."
        )

# Main validation function
def validate_prefix_feature_tensor(
    features: np.ndarray,
    channels: int,
    depth: int,
    scalar_term: bool,
    expected_rank: int | tuple[int, ...] = 2,
    name: str = "features"
) -> None:
    validate_finite_array(features, name = name)
    validate_feature_rank(features, expected_rank = expected_rank)
    validate_feature_dim_from_spec(features = features, channels = channels, depth = depth, scalar_term = scalar_term)

# Compare two feature tensors for deterministic rerun checks
def compare_feature_tensors(features_a: np.ndarray, features_b: np.ndarray, atol: float = 1e-7, rtol: float = 1e-6) -> None:
    if features_a.shape != features_b.shape:
        raise ValueError(
            f"Feature tensor shape mismatch: {features_a.shape} vs {features_b.shape}"
        )

    if not np.allclose(features_a, features_b, atol = atol, rtol = rtol):
        max_abs_diff = float(np.max(np.abs(features_a - features_b)))
        raise ValueError(
            f"Feature tensors differ beyond tolerance. max_abs_diff: {max_abs_diff}"
        )
