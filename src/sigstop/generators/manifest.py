from __future__ import annotations
import json
import os
import platform
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import numpy as np
from src.sigstop.features.manifest import compute_file_sha256, safe_package_version

# Summarize array into dictionary
def summarize_array(arr: np.ndarray) -> dict[str, Any]:
    if not np.all(np.isfinite(arr)):
        raise ValueError("Array contains NaN or infinite value(s)")

    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }

# Build OU sample manifest for logging
def build_ou_sample_manifest(
    *,
    pair: list[str],
    params: Any,
    sample_request: Any,
    paths: np.ndarray,
    innovations: np.ndarray | None = None,
    source_spread_path: str | Path,
    fit_diagnostics: Any | None = None,
    source_spread_hash: str | None = None,
    validation_report: Any | None = None,
    git_commit: str | None = None,
    array_path: str | Path | None = None,
    array_hash: str | None = None,
    fit_params_path: str | Path | None = None,
    fit_diagnostics_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_source_hash = source_spread_hash
    if resolved_source_hash is None:
        resolved_source_hash = compute_file_sha256(source_spread_path)

    resolved_array_hash = array_hash
    if resolved_array_hash is None and array_path is not None:
        resolved_array_hash = compute_file_sha256(array_path)

    manifest = {
        "generator_type": "ou",
        "pair": list(pair),
        "params": _serialize_payload(params),
        "sample_request": _serialize_payload(sample_request),
        "fit_diagnostics": _serialize_payload(fit_diagnostics),
        "provenance": {
            "source_spread_path": str(source_spread_path),
            "source_spread_hash": resolved_source_hash,
            "array_path": str(array_path) if array_path is not None else None,
            "array_hash": resolved_array_hash,
            "fit_params_path": str(fit_params_path) if fit_params_path is not None else None,
            "fit_diagnostics_path": (
                str(fit_diagnostics_path) if fit_diagnostics_path is not None else None
            ),
            "git_commit": git_commit,
        },
        "repro": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "os_name": os.name,
            "numpy_version": np.__version__,
            "torch_version": safe_package_version("torch"),
        },
        "array_facts": summarize_array(paths),
        "innovations_facts": (
            None if innovations is None else summarize_array(innovations)
        ),
        "validation": _serialize_payload(validation_report),
    }

    return manifest

# Save manifest into path
def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents = True, exist_ok = True)

    with open(path, "w", encoding = "utf-8") as handle:
        json.dump(manifest, handle, indent = 2)

# Turn into dictionary 
def _serialize_payload(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    return value
