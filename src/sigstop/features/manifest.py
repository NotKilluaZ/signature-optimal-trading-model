from __future__ import annotations
import hashlib
import json
import os
import platform
import sys
from importlib import metadata
from pathlib import Path
from typing import Any
import numpy as np

def safe_package_version(package_name: str) -> str | None:
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return None

def compute_file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    file_path = Path(path)
    digest = hashlib.sha256()

    with open(file_path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)

    return digest.hexdigest()

# Build compact summary of feature tensor
def summarize_array(features: np.ndarray) -> dict[str, Any]:
    if not np.all(np.isfinite(features)):
        raise ValueError("Feature tensor contains NaN or infinite value(s)")

    flat = features.ravel()
    n = flat.size

    # Process in small chunks so peak temporary memory stays ~2 MiB regardless of array size
    _CHUNK = 250_000
    lo = float(flat[0])
    hi = float(flat[0])
    s1 = 0.0
    s2 = 0.0
    for start in range(0, n, _CHUNK):
        chunk = flat[start : start + _CHUNK]
        lo = min(lo, float(chunk.min()))
        hi = max(hi, float(chunk.max()))
        s1 += float(np.sum(chunk, dtype = np.float64))
        s2 += float(np.sum(np.square(chunk), dtype = np.float64))
    mean = s1 / n
    std = float(np.sqrt(max(0.0, s2 / n - mean * mean)))
    return {
        "dtype": str(features.dtype),
        "shape": list(features.shape),
        "min": lo,
        "max": hi,
        "mean": mean,
        "std": std,
    }

# Main manifest
def build_feature_manifest(
    *,
    name: str,
    feature_spec: dict[str, Any],
    scaler_spec: dict[str, Any],
    features: np.ndarray,
    source_spread_path: str | Path | None = None,
    source_spread_hash: str | None = None,
    run_seed: int | None = None,
    git_commit: str | None = None,
    deterministic: bool | None = None,
    library_name: str = "esig",
    library_version: str | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    resolved_source_hash = source_spread_hash
    if resolved_source_hash is None and source_spread_path is not None:
        resolved_source_hash = compute_file_sha256(source_spread_path)

    resolved_library_version = library_version
    if resolved_library_version is None:
        resolved_library_version = safe_package_version(library_name)

    manifest = {
        "name": name,
        "feature_spec": feature_spec,
        "scaling": scaler_spec,
        "library": {
            "name": library_name,
            "version": resolved_library_version,
            "device": device,
        },
        "provenance": {
            "source_spread_path": str(source_spread_path) if source_spread_path else None,
            "source_spread_hash": resolved_source_hash,
            "run_seed": run_seed,
            "git_commit": git_commit,
        },
        "repro": {
            "deterministic": deterministic,
            "python_version": sys.version,
            "platform": platform.platform(),
            "os_name": os.name,
        },
        "array_facts": summarize_array(features),
    }

    return manifest

# Save manifest
def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    with open(path, "w", encoding = "utf-8") as handle:
        json.dump(manifest, handle, indent = 2)

# Load manifest
def load_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "r", encoding = "utf-8") as handle:
        return json.load(handle)
