from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import Any
import pandas as pd
from src.sigstop.config import load_config
from src.sigstop.generators.cache import build_ou_fit_paths, build_ou_sample_cache_paths, build_ou_validation_report_path, save_sample_cache
from src.sigstop.generators.manifest import build_ou_sample_manifest, save_manifest
from src.sigstop.generators.ou_fit import fit_ou_generator_params_from_artifact
from src.sigstop.generators.ou_sim import resolve_x0_from_spread_artifact, sample_ou_paths
from src.sigstop.generators.spec import GeneratorContext, build_default_sample_request, build_generator_context
from src.sigstop.generators.validation import build_ou_validation_report
from src.sigstop.paths import ensure_directories

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Save JSON to input path directory
def save_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents = True, exist_ok = True)
    with open(path, "w", encoding = "utf-8") as handle:
        json.dump(data, handle, indent = 2)

# Log OU fit summary
def build_fit_summary(
    context: GeneratorContext,
    *,
    params_path: Path,
    diagnostics_path: Path,
    sample_path: Path,
    sample_manifest_path: Path,
    validation_report_path: Path,
    sample_request: dict[str, Any],
    sample_batch: dict[str, Any],
    sample_validation: dict[str, Any],
) -> dict[str, Any]:
    return {
        "status": "sample_cached",
        "generator_type": "ou",
        "pair": context.pair,
        "inputs": {
            "formation_spread_path": str(context.formation_spread_path),
            "trading_spread_path": str(context.trading_spread_path),
            "selection_metadata_path": str(context.selection_metadata_path),
            "split_metadata_path": str(context.split_metadata_path),
            "source_spread_hash": context.source_spread_hash,
        },
        "fit_contract": {
            "method": context.estimation_method,
            "dt": context.dt,
            "window": "formation",
            "output_paths": {
                "params": str(params_path),
                "diagnostics": str(diagnostics_path),
            },
        },
        "sample_contract": {
            "default_horizon": context.horizon,
            "default_n_paths": context.n_paths,
            "dtype": context.dtype,
            "device": context.device,
            "include_innovations": context.include_innovations,
            "run_seed": context.run_seed,
            "output_paths": {
                "sample_npz": str(sample_path),
                "sample_manifest": str(sample_manifest_path),
                "validation_report": str(validation_report_path),
            },
        },
        "sample_artifacts": {
            "request": sample_request,
            "batch": sample_batch,
            "validation": sample_validation,
        },
        "validation_contract": {
            "enabled": context.validation_enabled,
        },
        "notes": [
            "Step 5 fits OU parameters, caches a sample batch, and writes a reproducibility manifest.",
            "Expanded diagnostics compare simulated OU behavior against exact transition formulas and the formation spread.",
        ],
    }

# Main orchastration function to load spread, fit OU params, generate OU sample paths, then validate results and log diagnostics / summaries
def main() -> None:
    # Load config values
    config = load_config()
    ensure_directories()

    context = build_generator_context(config)
    params_path, diagnostics_path = build_ou_fit_paths(context.synthetic_dir)
    var_floor = float(config["generator"]["fit"]["var_floor"])

    params, diagnostics = fit_ou_generator_params_from_artifact(
        context.formation_spread_path,
        dt = context.dt,
        var_floor = var_floor,
        method = context.estimation_method,
    )

    save_json(params.to_dict(), params_path)
    save_json(diagnostics.to_dict(), diagnostics_path)

    x0 = resolve_x0_from_spread_artifact(
        context.trading_spread_path,
        spread_column = "spread",
        row_index = 0,
    )
    sample_request = build_default_sample_request(
        x0 = x0,
        config = config,
        seed = context.run_seed,
    )
    sample_batch = sample_ou_paths(params, sample_request)
    formation_spread_df = pd.read_parquet(context.formation_spread_path)
    sample_validation = build_ou_validation_report(
        sample_batch.paths,
        params = params,
        request = sample_request,
        reference_spread = formation_spread_df["spread"].to_numpy(),
        acf_nlags = int(config["generator"]["validation"]["acf_nlags"]),
    )

    pair_name = "_".join(symbol.upper() for symbol in context.pair)
    sample_path, sample_manifest_path = build_ou_sample_cache_paths(
        context.synthetic_dir,
        pair_name = pair_name,
        n_paths = sample_request.n_paths,
        horizon = sample_request.horizon,
        dt = sample_request.dt,
        seed = sample_request.seed,
    )
    validation_report_path = build_ou_validation_report_path(
        context.synthetic_dir,
        pair_name = pair_name,
        n_paths = sample_request.n_paths,
        horizon = sample_request.horizon,
        dt = sample_request.dt,
        seed = sample_request.seed,
    )

    save_sample_cache(
        sample_path,
        paths = sample_batch.paths,
        innovations = sample_batch.innovations,
    )
    save_json(sample_validation.to_dict(), validation_report_path)

    sample_manifest = build_ou_sample_manifest(
        pair = context.pair,
        params = params,
        sample_request = sample_request,
        paths = sample_batch.paths,
        innovations = sample_batch.innovations,
        source_spread_path = context.formation_spread_path,
        fit_diagnostics = diagnostics,
        source_spread_hash = context.source_spread_hash,
        validation_report = sample_validation,
        git_commit = None,
        array_path = sample_path,
        fit_params_path = params_path,
        fit_diagnostics_path = diagnostics_path,
    )
    save_manifest(sample_manifest, sample_manifest_path)

    fit_summary = build_fit_summary(
        context,
        params_path = params_path,
        diagnostics_path = diagnostics_path,
        sample_path = sample_path,
        sample_manifest_path = sample_manifest_path,
        validation_report_path = validation_report_path,
        sample_request = sample_request.to_dict(),
        sample_batch = sample_batch.to_dict(),
        sample_validation = sample_validation.to_dict(),
    )

    summary_path = context.synthetic_dir / "manifest_scaffold.json"
    save_json(fit_summary, summary_path)

    print("\nOU generator validation run complete.")
    print(f"Synthetic root: {context.synthetic_dir}")
    print(f"OU params saved to: {params_path}")
    print(f"OU fit diagnostics saved to: {diagnostics_path}")
    print(f"Sample cache saved to: {sample_path}")
    print(f"Sample manifest saved to: {sample_manifest_path}")
    print(f"Validation report saved to: {validation_report_path}")
    print(f"Run summary saved to: {summary_path}")
    print(f"Preview x0 (first trading spread): {sample_request.x0:.6f}")
    print(f"Preview sample shape: {sample_batch.paths.shape}")
    print(f"Validation status: {sample_validation.status}")
    print("Next implementation step: tighten thresholds and add tests or plots around the new diagnostics.")


if __name__ == "__main__":
    main()
