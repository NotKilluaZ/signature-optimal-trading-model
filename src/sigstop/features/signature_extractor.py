from __future__ import annotations
from importlib import metadata
from typing import Any
import esig
import numpy as np
import roughpy as rp

# Ensure data type is valid
def _resolve_dtype_name(dtype: str | np.dtype | type[np.floating]) -> str:
    resolved = np.dtype(dtype)
    if resolved == np.dtype(np.float32):
        return "float32"
    if resolved == np.dtype(np.float64):
        return "float64"
    raise ValueError(f"Unsupported dtype. Got: {resolved}")

# Ensure data type valid for numpy (fallback for esig)
def _resolve_numpy_dtype(dtype: str | np.dtype | type[np.floating]) -> np.dtype:
    dtype_name = _resolve_dtype_name(dtype)
    return np.dtype(np.float32 if dtype_name == "float32" else np.float64)

def _safe_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None

# Ensure esig version has required attributes / functions
def _get_esig_backend() -> Any:
    if hasattr(esig, "stream2sig") and hasattr(esig, "sigdim"):
        return esig

    if hasattr(esig, "tosig") and hasattr(esig.tosig, "stream2sig") and hasattr(esig.tosig, "sigdim"):
        return esig.tosig

    raise ImportError(
        "Installed esig package does not expose the expected signature API."
    )

# Validate requirements for inputted augmented paths 
def validate_augmented_paths(path: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(path, dtype = float)
    squeezed = False

    if x.ndim == 2:
        x = x[None, :, :]
        squeezed = True
    elif x.ndim != 3:
        raise ValueError(
            "Expected augmented path with shape (L, C) or (B, L, C). "
            f"Got shape: {x.shape}"
        )

    if x.shape[1] < 2:
        raise ValueError(
            "Augmented path must contain at least 2 observations. "
            f"Got length: {x.shape[1]}"
        )
    if x.shape[2] < 1:
        raise ValueError(
            "Augmented path must have at least 1 channel. "
            f"Got channels: {x.shape[2]}"
        )
    if not np.all(np.isfinite(x)):
        raise ValueError("Augmented path contains NaN or infinite value(s).")

    return x, squeezed

# Calucate expected signature dimension to compare with calculated signature
def expected_signature_dim(channels: int, depth: int, scalar_term: bool = True) -> int:
    if channels < 1:
        raise ValueError(f"Channels must be greater than or equal to 1. Got: {channels}")
    if depth < 1:
        raise ValueError(f"Depth must be at least 1. Got: {depth}")

    dim = sum(channels ** k for k in range(1, depth + 1))

    if scalar_term:
        dim += 1

    return dim

# Build prefixes
def build_prefix_ends(
    path_length: int,
    min_prefix: int = 2,
    prefix_ends: np.ndarray | list[int] | None = None
) -> np.ndarray:
    if min_prefix < 2:
        raise ValueError(f"min_prefix must be at least 2. Got: {min_prefix}")
    if path_length < min_prefix:
        raise ValueError(
            f"Path length {path_length} is too short for min_prefix={min_prefix}."
        )

    min_end = min_prefix - 1

    if prefix_ends is None:
        return np.arange(min_end, path_length, dtype = np.int32)

    ends = np.asarray(prefix_ends, dtype = np.int32)

    if ends.ndim != 1 or ends.size == 0:
        raise ValueError("prefix_ends must be a non-empty 1D array of indices.")
    if np.any(ends < min_end) or np.any(ends >= path_length):
        raise ValueError(
            "prefix_ends contains invalid indices. "
            f"Allowed range is [{min_end}, {path_length - 1}]."
        )
    if np.any(np.diff(ends) <= 0):
        raise ValueError("prefix_ends must be strictly increasing.")

    return ends

# Check if esig is available
def _resolve_library(library: str) -> str:
    if library != "esig":
        raise ValueError(
            "This project is configured to use only the esig backend. "
            f"Got library={library!r}"
        )
    return "esig"

# Check if device set to GPU. Can only use CPU after switching to esig :(
def _resolve_device(device: str) -> str:
    if device != "cpu":
        raise ValueError(
            "The esig backend is configured for CPU execution only. "
            f"Got device={device!r}"
        )
    return "cpu"

def _prepare_esig_prefix_path(path_prefix: np.ndarray, basepoint: bool) -> np.ndarray:
    if not basepoint:
        return path_prefix

    zero_row = np.zeros((1, path_prefix.shape[1]), dtype = path_prefix.dtype)
    return np.vstack([zero_row, path_prefix])

# Use RoughPy to compute signature using it's streaming function for efficient prefix accumulation
def _compute_signature_with_roughpy(stream: np.ndarray, depth: int) -> np.ndarray:
    no_samples, width = stream.shape
    expected_dim = expected_signature_dim(width, depth, scalar_term = True)

    if no_samples == 1:
        signature = np.zeros(expected_dim, dtype = np.float64)
        signature[0] = 1.0
        return signature

    increments = np.diff(stream, axis = 0)

    # esig 1.0.0 builds indices with np.arange(0, 1, step), which can
    # produce one extra index for some prefix lengths due to rounding.
    indices = np.arange(increments.shape[0], dtype = np.float64) / (no_samples - 1)

    context = rp.get_context(width, depth, rp.DPReal)
    lie_stream = rp.LieIncrementStream.from_increments(
        increments,
        indices = indices,
        ctx = context,
    )

    return np.array(
        lie_stream.signature(rp.RealInterval(0.0, 1.0)),
        copy = True,
    )

# Compute prefix signatures with esig (call the RoughPy streaming helper for each prefix)
def _compute_prefix_signatures_esig(
    batched_paths: np.ndarray,
    depth: int,
    prefix_ends: np.ndarray,
    scalar_term: bool,
    basepoint: bool
) -> np.ndarray:
    ts = _get_esig_backend()
    channels = batched_paths.shape[2]
    esig_dim = int(ts.sigdim(channels, depth))
    expected_dim = expected_signature_dim(channels, depth, scalar_term)

    if esig_dim != expected_signature_dim(channels, depth, scalar_term = True):
        raise ValueError(
            "esig signature dimension does not match the expected tensor dimension. "
            f"esig sigdim={esig_dim}"
        )

    all_features = np.empty(
        (batched_paths.shape[0], len(prefix_ends), expected_dim),
        dtype = np.float64,
    )

    for batch_index in range(batched_paths.shape[0]):
        for prefix_index, prefix_end in enumerate(prefix_ends):
            prefix_path = batched_paths[batch_index, : prefix_end + 1, :]
            esig_path = _prepare_esig_prefix_path(prefix_path, basepoint = basepoint)
            sig = _compute_signature_with_roughpy(esig_path, depth)

            if sig.ndim != 1 or sig.shape[0] != esig_dim:
                raise ValueError(
                    "esig returned a signature with unexpected shape. "
                    f"Got shape: {sig.shape}, expected length: {esig_dim}"
                )

            all_features[batch_index, prefix_index] = sig if scalar_term else sig[1:]

    return all_features

# Main function that combines all helpers to calculate final prefix signatures and log details
def compute_prefix_signatures(
    augmented_path: np.ndarray,
    depth: int,
    *,
    scalar_term: bool = True,
    dtype: str | np.dtype | type[np.floating] = "float32",
    device: str = "cpu",
    basepoint: bool = False,
    min_prefix: int = 2,
    prefix_ends: np.ndarray | list[int] | None = None,
    mode: str = "prefix",
    library: str = "esig"
) -> tuple[np.ndarray, dict[str, Any]]:
    if mode not in {"prefix", "full"}:
        raise ValueError(f"Unsupported mode. Expected 'prefix' or 'full', got: {mode}")

    batched_paths, squeezed = validate_augmented_paths(augmented_path)
    dtype_name = _resolve_dtype_name(dtype)
    np_dtype = _resolve_numpy_dtype(dtype)
    resolved_prefix_ends = build_prefix_ends(
        path_length = batched_paths.shape[1],
        min_prefix = min_prefix,
        prefix_ends = prefix_ends,
    )
    resolved_library = _resolve_library(library)
    resolved_device = _resolve_device(device)

    feature_array = _compute_prefix_signatures_esig(
        batched_paths = batched_paths,
        depth = depth,
        prefix_ends = resolved_prefix_ends,
        scalar_term = scalar_term,
        basepoint = basepoint,
    ).astype(np_dtype, copy = False)
    library_version = _safe_version("esig")

    channels = batched_paths.shape[2]
    expected_dim = expected_signature_dim(channels, depth, scalar_term)

    if feature_array.ndim != 3:
        raise ValueError(
            "Prefix signature output must have shape (B, P, D). "
            f"Got shape: {feature_array.shape}"
        )
    if feature_array.shape[-1] != expected_dim:
        raise ValueError(
            f"Prefix signature dimension mismatch. Expected: {expected_dim}, "
            f"got: {feature_array.shape[-1]}"
        )

    if mode == "full":
        output = feature_array[:, -1, :]
    else:
        output = feature_array

    if squeezed:
        output = output[0]

    feature_spec = {
        "library": resolved_library,
        "library_version": library_version,
        "mode": mode,
        "input_shape": list(np.asarray(augmented_path).shape),
        "batched_input_shape": list(batched_paths.shape),
        "output_shape": list(output.shape),
        "channels": int(channels),
        "depth": int(depth),
        "scalar_term": bool(scalar_term),
        "dtype": dtype_name,
        "device": resolved_device,
        "basepoint": bool(basepoint),
        "min_prefix": int(min_prefix),
        "prefix_index_convention": (
            "Each output row corresponds to the signature of the original path "
            "prefix ending at the same index in prefix_ends."
        ),
        "prefix_ends": resolved_prefix_ends.tolist(),
        "expected_dim": int(expected_dim)
    }

    return output, feature_spec
