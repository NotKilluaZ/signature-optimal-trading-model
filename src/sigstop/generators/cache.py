from __future__ import annotations
from pathlib import Path
import numpy as np

# Ensure directories exist
def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)

# Ensure directory paths for artifacts
def build_ou_fit_paths(base_dir: str | Path) -> tuple[Path, Path]:
    base_dir = Path(base_dir)
    fit_dir = base_dir / "fit"
    params_path = fit_dir / "ou_params_formation.json"
    diagnostics_path = fit_dir / "ou_fit_diagnostics.json"
    return params_path, diagnostics_path

# Ensure directory paths for ou path sample cache
def build_ou_sample_cache_paths(
    base_dir: str | Path,
    *,
    pair_name: str,
    n_paths: int,
    horizon: int,
    dt: float,
    seed: int,
    chunk_id: int | None = None,
) -> tuple[Path, Path]:
    base_dir = Path(base_dir)
    samples_dir = base_dir / "samples"
    manifests_dir = base_dir / "manifests"

    dt_label = _format_dt_label(dt)
    stem = f"{pair_name}_ou_M{n_paths}_H{horizon}_dt{dt_label}_seed{seed}"
    if chunk_id is not None:
        stem = f"{stem}_chunk{chunk_id:03d}"

    feature_path = samples_dir / f"{stem}.npz"
    manifest_path = manifests_dir / f"manifest_{stem}.json"
    return feature_path, manifest_path

# Validation report directory
def build_ou_validation_report_path(
    base_dir: str | Path,
    *,
    pair_name: str,
    n_paths: int,
    horizon: int,
    dt: float,
    seed: int,
    chunk_id: int | None = None,
) -> Path:
    base_dir = Path(base_dir)
    validation_dir = base_dir / "validation"

    dt_label = _format_dt_label(dt)
    stem = f"{pair_name}_ou_M{n_paths}_H{horizon}_dt{dt_label}_seed{seed}"
    if chunk_id is not None:
        stem = f"{stem}_chunk{chunk_id:03d}"

    return validation_dir / f"validation_{stem}.json"

# Save sample path into cache
def save_sample_cache(
    sample_path: str | Path,
    *,
    paths: np.ndarray,
    innovations: np.ndarray | None = None,
) -> None:
    sample_path = Path(sample_path)
    ensure_parent_dir(sample_path)

    if not np.all(np.isfinite(paths)):
        raise ValueError("Sample paths contain NaN or infinite value(s)")

    if innovations is not None:
        if not np.all(np.isfinite(innovations)):
            raise ValueError("Innovations contain NaN or infinite value(s)")
        np.savez(sample_path, paths = paths, innovations = innovations)
        return

    np.savez(sample_path, paths = paths)

# Load saved sample path from cache
def load_sample_cache(sample_path: str | Path) -> dict[str, np.ndarray]:
    sample_path = Path(sample_path)

    if not sample_path.exists():
        raise FileNotFoundError(f"OU sample cache not found: {sample_path}")

    with np.load(sample_path, allow_pickle = False) as data:
        return {key: data[key] for key in data.files}

# Format data type 
def _format_dt_label(dt: float) -> str:
    integer_dt = int(dt)
    if float(integer_dt) == float(dt):
        return str(integer_dt)
    return str(dt).replace(".", "p")
