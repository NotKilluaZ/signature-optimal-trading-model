from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from src.sigstop.features.manifest import compute_file_sha256
from src.sigstop.paths import ROOT_DIR
from .types import OUSampleRequest


@dataclass(frozen = True)
class GeneratorContext:
    pair: list[str]
    synthetic_dir: Path
    formation_spread_path: Path
    trading_spread_path: Path
    selection_metadata_path: Path
    split_metadata_path: Path
    source_spread_hash: str
    dt: float
    horizon: int
    n_paths: int
    dtype: str
    device: str
    include_innovations: bool
    run_seed: int
    estimation_method: str
    validation_enabled: bool

# Load variables from config file YAML
def build_generator_context(config: dict[str, Any]) -> GeneratorContext:
    source_artifacts_dir = ROOT_DIR / _get_config_value(
        config,
        ["features", "source_artifacts_dir"],
        default = "data/processed/artifacts",
    )
    synthetic_dir = ROOT_DIR / _get_config_value(
        config,
        ["generator", "cache", "synthetic_dir"],
        default = "data/synthetic/gs_ms/ou",
    )

    formation_spread_path = source_artifacts_dir / "spread_formation.parquet"
    trading_spread_path = source_artifacts_dir / "spread_trading.parquet"
    selection_metadata_path = source_artifacts_dir / "spread_selection_data.json"
    split_metadata_path = ROOT_DIR / "data" / "processed" / "split_metadata.json"

    dt = float(_get_config_value(config, ["generator", "fit", "dt"], default = 1.0))
    horizon = int(_get_config_value(config, ["generator", "sample", "horizon"], default = 252))
    n_paths = int(_get_config_value(config, ["generator", "sample", "n_paths"], default = 2000))
    dtype = str(_get_config_value(config, ["generator", "sample", "dtype"], default = "float64"))
    device = str(_get_config_value(config, ["generator", "sample", "device"], default = "cpu"))
    include_innovations = bool(
        _get_config_value(config, ["generator", "sample", "include_innovations"], default = False)
    )
    run_seed = int(_get_config_value(config, ["repro", "seed"], default = 42))
    estimation_method = str(
        _get_config_value(config, ["generator", "fit", "method"], default = "transition_mle")
    )
    validation_enabled = bool(
        _get_config_value(config, ["generator", "validation", "enabled"], default = True)
    )
    pair = list(_get_config_value(config, ["data", "symbols"], default = ["GS", "MS"]))

    source_spread_hash = compute_file_sha256(formation_spread_path)

    return GeneratorContext(
        pair = pair,
        synthetic_dir = synthetic_dir,
        formation_spread_path = formation_spread_path,
        trading_spread_path = trading_spread_path,
        selection_metadata_path = selection_metadata_path,
        split_metadata_path = split_metadata_path,
        source_spread_hash = source_spread_hash,
        dt = dt,
        horizon = horizon,
        n_paths = n_paths,
        dtype = dtype,
        device = device,
        include_innovations = include_innovations,
        run_seed = run_seed,
        estimation_method = estimation_method,
        validation_enabled = validation_enabled,
    )


def build_default_sample_request(
    *,
    x0: float,
    config: dict[str, Any],
    seed: int | None = None,
) -> OUSampleRequest:
    resolved_seed = int(
        seed if seed is not None else _get_config_value(config, ["repro", "seed"], default = 42)
    )

    return OUSampleRequest(
        x0 = float(x0),
        horizon = int(_get_config_value(config, ["generator", "sample", "horizon"], default = 252)),
        n_paths = int(_get_config_value(config, ["generator", "sample", "n_paths"], default = 2000)),
        seed = resolved_seed,
        dt = float(_get_config_value(config, ["generator", "fit", "dt"], default = 1.0)),
        dtype = str(_get_config_value(config, ["generator", "sample", "dtype"], default = "float64")),
        device = str(_get_config_value(config, ["generator", "sample", "device"], default = "cpu")),
        include_innovations = bool(
            _get_config_value(config, ["generator", "sample", "include_innovations"], default = False)
        ),
    )


def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current
