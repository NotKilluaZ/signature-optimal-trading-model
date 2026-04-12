from __future__ import annotations
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from src.sigstop.backtest.engine import BacktestDayContext, BacktestStrategy, StrategyAction, StrategyDecision
from src.sigstop.config import load_config
from src.sigstop.features.cache import build_backtest_feature_cache_inputs, build_backtest_feature_cache_key, build_backtest_feature_cache_paths, load_feature_build_result, save_feature_build_result
from src.sigstop.features.feature_builder import FeatureBuildResult, build_feature_tensor
from src.sigstop.features.manifest import build_feature_manifest, save_manifest
from src.sigstop.features.scaling import fit_zscore_scaler, scaler_to_dict
from src.sigstop.generators.cache import build_ou_fit_paths
from src.sigstop.generators.ou_fit import fit_ou_generator_params
from src.sigstop.generators.ou_sim import sample_ou_paths
from src.sigstop.generators.spec import build_default_sample_request, build_generator_context
from src.sigstop.generators.types import OUGeneratorParams, OUSampleRequest
from src.sigstop.paths import ROOT_DIR, RUNS_DIR, ensure_directories
from src.sigstop.stopping.policy import score_signature_features
from src.sigstop.stopping.stop_rule import apply_deterministic_stop_rule
from src.sigstop.train.train_entry import build_entry_training_data, train_entry_policy
from src.sigstop.train.train_exit import build_exit_training_data, train_exit_policy
from src.sigstop.train.trainer import (
    TrainingResult,
    build_default_run_id,
    build_training_config_from_dict,
    train_linear_stopping_policy,
)


@dataclass
class SOTStagePlan:
    stage: str
    state_start_index: int
    state_start_date: str | None
    initial_spread: float
    horizon: int
    threshold: float
    output_dir: Path
    policy_id: str
    training_result: Any | None
    prefix_ends: np.ndarray
    scores: np.ndarray
    cumulative_squared_scores: np.ndarray
    trigger_prefix_index: int | None
    trigger_row_index: int | None
    trigger_date: str | None
    trigger_spread: float | None
    plan_start_index: int = 0
    feature_spec: dict[str, Any] = field(default_factory = dict)
    feature_cache_key: str | None = None
    feature_cache_path: Path | None = None
    feature_manifest_path: Path | None = None
    feature_cache_hit: bool = False

    # Get the latest prefix index available at the current day
    def latest_available_prefix_index(self, current_index: int) -> int | None:
        relative_index = int(current_index - self.plan_start_index)
        if relative_index < 0 or self.prefix_ends.size == 0:
            return None

        available = np.flatnonzero(self.prefix_ends <= relative_index)
        if available.size == 0:
            return None
        return int(available[-1])


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Write a nested config value while creating missing dictionaries
def _set_config_value(config: dict[str, Any], path: list[str], value: Any) -> None:
    current = config
    for key in path[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[path[-1]] = value


# Serialize a date-like value into ISO day format
def _serialize_date(value: Any) -> str | None:
    if value is None:
        return None
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return None
    return str(timestamp.date())


# Overlay the SOT backtest defaults onto the full config
def build_effective_sot_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = deepcopy(config)

    sot_k = float(
        _get_config_value(
            resolved,
            ["sot", "k"],
            _get_config_value(resolved, ["stopping", "deterministic_threshold", "k"], 0.05),
        )
    )
    sot_mu = float(
        _get_config_value(
            resolved,
            ["sot", "mu"],
            _get_config_value(resolved, ["stopping", "deterministic_threshold", "mu"], 20.0),
        )
    )
    sot_samples = int(
        _get_config_value(
            resolved,
            ["sot", "ou_samples_M"],
            _get_config_value(resolved, ["generator", "sample", "n_paths"], 2000),
        )
    )
    sot_depth = int(
        _get_config_value(
            resolved,
            ["sot", "feature_depth_N"],
            _get_config_value(resolved, ["features", "signature", "depth"], 4),
        )
    )

    _set_config_value(resolved, ["sot", "k"], sot_k)
    _set_config_value(resolved, ["sot", "mu"], sot_mu)
    _set_config_value(resolved, ["sot", "ou_samples_M"], sot_samples)
    _set_config_value(resolved, ["sot", "feature_depth_N"], sot_depth)

    _set_config_value(resolved, ["stopping", "deterministic_threshold", "k"], sot_k)
    _set_config_value(resolved, ["stopping", "deterministic_threshold", "mu"], sot_mu)
    _set_config_value(resolved, ["training", "M"], sot_samples)
    _set_config_value(resolved, ["generator", "sample", "n_paths"], sot_samples)
    _set_config_value(resolved, ["features", "signature", "depth"], sot_depth)
    return resolved


# Resolve the persistent root directory for cached backtest stage features
def resolve_backtest_feature_cache_root(config: dict[str, Any]) -> Path:
    features_dir = Path(_get_config_value(config, ["features", "artifacts_dir"], "data/features"))
    if not features_dir.is_absolute():
        features_dir = ROOT_DIR / features_dir
    return features_dir / "backtest_stage_cache"


# Resolve the persistent root directory for cached synthetic SOT training tensors
def resolve_sot_training_cache_root(config: dict[str, Any]) -> Path:
    synthetic_dir = Path(
        _get_config_value(config, ["generator", "cache", "synthetic_dir"], "data/synthetic/gs_ms/ou")
    )
    if not synthetic_dir.is_absolute():
        synthetic_dir = ROOT_DIR / synthetic_dir
    return synthetic_dir


# Build the feature settings dictionary used by real SOT stage tensors
def build_real_stage_feature_settings(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "depth": int(_get_config_value(config, ["features", "signature", "depth"], 4)),
        "min_prefix": int(_get_config_value(config, ["features", "signature", "min_prefix"], 2)),
        "scalar_term": bool(_get_config_value(config, ["features", "signature", "scalar_term"], True)),
        "dtype": str(_get_config_value(config, ["features", "signature", "dtype"], "float32")),
        "time_representation": str(
            _get_config_value(config, ["features", "augmentation", "time_representation"], "index_0_1")
        ),
        "library": str(_get_config_value(config, ["features", "signature", "library"], "esig")),
        "device": str(_get_config_value(config, ["features", "signature", "device"], "cpu")),
        "basepoint": bool(_get_config_value(config, ["features", "signature", "basepoint"], False)),
        "mode": "prefix",
    }


# Hash one numeric array into a stable digest for runtime cache provenance
def _hash_numeric_array(values: np.ndarray) -> str:
    resolved = np.ascontiguousarray(np.asarray(values, dtype = np.float64))
    digest = hashlib.sha256()
    digest.update(str(resolved.dtype).encode("utf-8"))
    digest.update(np.asarray(resolved.shape, dtype = np.int64).tobytes())
    digest.update(resolved.view(np.uint8).tobytes())
    return digest.hexdigest()


# Serialize one sample-request-like object into stable cache metadata
def _serialize_sample_request(sample_request: Any) -> dict[str, Any]:
    if hasattr(sample_request, "to_dict"):
        payload = sample_request.to_dict()
        if isinstance(payload, dict):
            return payload
    if hasattr(sample_request, "__dict__"):
        return {
            str(key): value
            for key, value in vars(sample_request).items()
            if not str(key).startswith("_")
        }
    raise TypeError("sample_request must provide to_dict() or __dict__ for cache serialization.")


# Build deterministic source metadata for one cached SOT stage training dataset
def build_sot_stage_training_cache_source(
    *,
    stage: str,
    sample_request: OUSampleRequest,
    ou_params: OUGeneratorParams,
    formation_spread: np.ndarray,
) -> dict[str, Any]:
    return {
        "cache_kind": "sot_runtime_ou_sample",
        "stage": str(stage),
        "sample_request": _serialize_sample_request(sample_request),
        "ou_params": ou_params.to_dict(),
        "formation_spread_hash": _hash_numeric_array(formation_spread),
    }


# Load the formation spread used to scale SOT real-path features
def load_sot_formation_spread(config: dict[str, Any]) -> np.ndarray:
    context = build_generator_context(config)
    formation_df = pd.read_parquet(context.formation_spread_path)
    if "spread" not in formation_df.columns:
        raise ValueError(
            f"Expected spread column 'spread' in formation artifact: {context.formation_spread_path}"
        )
    return formation_df["spread"].to_numpy(dtype = np.float64)


# Load the fitted OU generator parameters used for SOT sampling
def load_sot_ou_generator_params(config: dict[str, Any]) -> OUGeneratorParams:
    context = build_generator_context(config)
    params_path, _ = build_ou_fit_paths(context.synthetic_dir)

    with open(params_path, "r", encoding = "utf-8") as handle:
        payload = json.load(handle)
    return OUGeneratorParams(
        kappa = float(payload["kappa"]),
        theta = float(payload["theta"]),
        sigma = float(payload["sigma"]),
        dt = float(payload["dt"]),
        phi = float(payload["phi"]),
        half_life_days = float(payload["half_life_days"]),
        estimation_method = str(payload["estimation_method"]),
    )


# Build real-path prefix features for one SOT entry or exit stage
def build_real_stage_feature_tensor(
    spread_segment: np.ndarray,
    formation_spread: np.ndarray,
    config: dict[str, Any],
) -> FeatureBuildResult:
    feature_settings = build_real_stage_feature_settings(config)
    scaler = fit_zscore_scaler(formation_spread)
    return build_feature_tensor(
        spread_segment,
        scaler,
        depth = int(feature_settings["depth"]),
        min_prefix = int(feature_settings["min_prefix"]),
        scalar_term = bool(feature_settings["scalar_term"]),
        dtype = str(feature_settings["dtype"]),
        time_representation = str(feature_settings["time_representation"]),
        library = str(feature_settings["library"]),
        device = str(feature_settings["device"]),
        basepoint = bool(feature_settings["basepoint"]),
        mode = str(feature_settings["mode"]),
    )


# Sample synthetic OU paths for one SOT entry or exit stage
def sample_sot_stage_paths(
    config: dict[str, Any],
    ou_params: OUGeneratorParams,
    *,
    x0: float,
    horizon: int,
    seed: int | None = None,
) -> tuple[np.ndarray, OUSampleRequest]:
    default_request = build_default_sample_request(x0 = x0, config = config, seed = seed)
    request = OUSampleRequest(
        x0 = float(x0),
        horizon = int(horizon),
        n_paths = int(default_request.n_paths),
        seed = int(default_request.seed),
        dt = float(default_request.dt),
        dtype = str(default_request.dtype),
        device = str(default_request.device),
        include_innovations = bool(default_request.include_innovations),
    )
    sample_batch = sample_ou_paths(ou_params, request)
    return sample_batch.paths, request


class SOTBacktestStrategy(BacktestStrategy):
    name = "sot"

    # Initialize the SOT backtest strategy from config and artifacts
    def __init__(
        self,
        *,
        config_path: str | Path | None = None,
        config: dict[str, Any] | None = None,
        formation_spread: np.ndarray | None = None,
        ou_params: OUGeneratorParams | None = None,
        output_root: str | Path | None = None,
        run_id: str | None = None,
    ) -> None:
        if config is not None and config_path is not None:
            raise ValueError("Provide either config or config_path, not both.")

        base_config = (
            dict(config)
            if config is not None
            else (load_config(config_path) if config_path is not None else load_config())
        )
        self.config = build_effective_sot_config(base_config)
        ensure_directories()

        self.run_id = run_id or build_default_run_id(prefix = "sot_backtest")
        self.output_root = (
            Path(output_root)
            if output_root is not None
            else RUNS_DIR / self.run_id / "backtest" / "sot"
        )

        self.formation_spread = (
            np.asarray(formation_spread, dtype = np.float64)
            if formation_spread is not None
            else load_sot_formation_spread(self.config)
        )
        self.ou_params = ou_params if ou_params is not None else load_sot_ou_generator_params(self.config)
        self.threshold = float(
            _get_config_value(
                self.config,
                ["sot", "k"],
                _get_config_value(self.config, ["stopping", "deterministic_threshold", "k"], 0.05),
            )
        )
        self.mu = float(
            _get_config_value(
                self.config,
                ["sot", "mu"],
                _get_config_value(self.config, ["stopping", "deterministic_threshold", "mu"], 20.0),
            )
        )
        self.cache_episode_features = bool(
            _get_config_value(self.config, ["sot", "cache_episode_features"], True)
        )
        self.rolling_refit_enabled = bool(
            _get_config_value(self.config, ["sot", "rolling_refit", "enabled"], False)
        )
        self.rolling_refit_window = int(
            _get_config_value(self.config, ["sot", "rolling_refit", "window"], 252)
        )
        self.ou_fit_dt = float(_get_config_value(self.config, ["generator", "fit", "dt"], 1.0))
        self.ou_fit_var_floor = float(
            _get_config_value(self.config, ["generator", "fit", "var_floor"], 1e-12)
        )
        self.ou_fit_method = str(
            _get_config_value(self.config, ["generator", "fit", "method"], "transition_mle")
        )
        self.feature_cache_root = resolve_backtest_feature_cache_root(self.config)
        self.training_cache_root = resolve_sot_training_cache_root(self.config)

        self._active_entry_plan: SOTStagePlan | None = None
        self._active_exit_plan: SOTStagePlan | None = None
        self.entry_stage_history: list[SOTStagePlan] = []
        self.exit_stage_history: list[SOTStagePlan] = []

        self.pretrain_horizon = int(
            _get_config_value(
                self.config,
                ["sot", "pretrain_horizon"],
                _get_config_value(
                    self.config,
                    ["training", "horizon_days"],
                    _get_config_value(self.config, ["sot", "episode_horizon"], 252),
                ),
            )
        )
        if self.pretrain_horizon <= 0:
            self.pretrain_horizon = 252
        self.pretrain_x0_count = int(
            _get_config_value(self.config, ["sot", "pretrain_x0_count"], 16)
        )
        self.pretrained_entry_result: TrainingResult | None = None
        self.pretrained_exit_result: TrainingResult | None = None
        self._pretrain_global_policies()

    # Sample a single-x0 bank of synthetic OU paths used to pretrain a policy
    def _sample_diversified_training_paths(self, *, stage: str) -> np.ndarray:
        formation = np.asarray(self.formation_spread, dtype = np.float64)
        x0 = float(formation[-1]) if formation.size > 0 else float(self.ou_params.theta)

        n_paths = int(
            _get_config_value(
                self.config,
                ["sot", "ou_samples_M"],
                _get_config_value(self.config, ["generator", "sample", "n_paths"], 100),
            )
        )
        n_paths = max(n_paths, 1)

        base_seed = int(_get_config_value(self.config, ["repro", "seed"], 42))
        slot_seed = int(
            hashlib.sha256(f"{base_seed}:pretrain:{stage}".encode()).hexdigest()[:8], 16
        ) % (2**31)
        default_request = build_default_sample_request(
            x0 = x0,
            config = self.config,
            seed = slot_seed,
        )
        request = OUSampleRequest(
            x0 = x0,
            horizon = int(self.pretrain_horizon),
            n_paths = n_paths,
            seed = slot_seed,
            dt = float(default_request.dt),
            dtype = str(default_request.dtype),
            device = str(default_request.device),
            include_innovations = False,
        )
        batch = sample_ou_paths(self.ou_params, request)
        return np.asarray(batch.paths, dtype = np.float64)

    # Pretrain one global entry policy and one global exit policy
    def _pretrain_global_policies(self) -> None:
        training_config = build_training_config_from_dict(self.config)

        entry_paths = self._sample_diversified_training_paths(stage = "entry")
        entry_data = build_entry_training_data(entry_paths, self.formation_spread, self.config)
        entry_output_dir = self.output_root / "pretrain" / "entry"
        self.pretrained_entry_result = train_linear_stopping_policy(
            entry_data.features,
            entry_data.payoffs,
            training_config,
            output_dir = entry_output_dir,
            stage = "entry",
            run_id = self.run_id,
            extra_manifest_data = {
                "source": "pretrain_diversified_x0",
                "horizon": int(self.pretrain_horizon),
                "n_paths": int(entry_paths.shape[0]),
                "feature_spec": entry_data.feature_spec,
                "scaler_spec": entry_data.scaler_spec,
            },
        )

        exit_paths = self._sample_diversified_training_paths(stage = "exit")
        exit_data = build_exit_training_data(exit_paths, self.formation_spread, self.config)
        exit_output_dir = self.output_root / "pretrain" / "exit"
        self.pretrained_exit_result = train_linear_stopping_policy(
            exit_data.features,
            exit_data.payoffs,
            training_config,
            output_dir = exit_output_dir,
            stage = "exit",
            run_id = self.run_id,
            extra_manifest_data = {
                "source": "pretrain_diversified_x0",
                "horizon": int(self.pretrain_horizon),
                "n_paths": int(exit_paths.shape[0]),
                "feature_spec": exit_data.feature_spec,
                "scaler_spec": exit_data.scaler_spec,
            },
        )

    # Return True when the cached plan has aged past its episode horizon
    def _plan_is_stale(self, plan: SOTStagePlan | None, context: BacktestDayContext) -> bool:
        if plan is None:
            return True
        if plan.state_start_index != context.state_start_index:
            return True
        if plan.horizon <= 0:
            return True
        if int(context.current_index) > int(plan.plan_start_index) + int(plan.horizon):
            return True
        return False

    # Evaluate the SOT entry stage while flat
    def on_flat_day(self, context: BacktestDayContext) -> StrategyDecision:
        self._active_exit_plan = None
        plan = self._active_entry_plan
        if self._plan_is_stale(plan, context):
            plan = self._build_stage_plan("entry", context)
            self._active_entry_plan = plan
            self.entry_stage_history.append(plan)

        decision = self._build_stage_decision(plan, context)
        if decision.action == StrategyAction.ENTER_LONG_SPREAD:
            self._active_entry_plan = None
        return decision

    # Evaluate the SOT exit stage while long spread
    def on_position_day(self, context: BacktestDayContext) -> StrategyDecision:
        if context.open_position is None:
            raise ValueError("LONG_SPREAD state requires an open_position.")

        self._active_entry_plan = None
        plan = self._active_exit_plan
        if self._plan_is_stale(plan, context):
            plan = self._build_stage_plan("exit", context)
            self._active_exit_plan = plan
            self.exit_stage_history.append(plan)

        decision = self._build_stage_decision(plan, context)
        if decision.action == StrategyAction.EXIT_LONG_SPREAD:
            self._active_exit_plan = None
        return decision

    # Build a rolling refit history ending just before the given stage start,
    # drawn from the pre-trading formation artifact plus trading days already
    # observed in this backtest. Returns None when rolling refit is disabled
    def _build_rolling_history(
        self,
        *,
        trading_window: pd.DataFrame,
        state_start_index: int,
    ) -> np.ndarray | None:
        if not self.rolling_refit_enabled:
            return None
        trading_prefix = (
            trading_window["spread"].to_numpy(dtype = np.float64)[:state_start_index]
            if state_start_index > 0
            else np.empty(0, dtype = np.float64)
        )
        combined = np.concatenate([self.formation_spread, trading_prefix])
        window = int(self.rolling_refit_window)
        if window > 0 and combined.size > window:
            combined = combined[-window:]
        return combined

    # Refit the OU generator parameters on a rolling history, falling back to
    # the pre-trading OU fit if the refit fails or the history is too short.
    def _refit_ou_params(self, history: np.ndarray) -> OUGeneratorParams:
        if history.size < 3:
            return self.ou_params
        try:
            params, _ = fit_ou_generator_params(
                pd.Series(history),
                dt = self.ou_fit_dt,
                var_floor = self.ou_fit_var_floor,
                method = self.ou_fit_method,
            )
        except (ValueError, RuntimeError):
            return self.ou_params
        return params

    # Reuse or build the cached real-path feature tensor for one SOT stage
    def _load_or_build_stage_feature_result(
        self,
        *,
        stage: str,
        stage_start_index: int,
        spread_segment: np.ndarray,
        formation_spread: np.ndarray,
    ) -> tuple[FeatureBuildResult, str | None, Path | None, Path | None, bool]:
        feature_settings = build_real_stage_feature_settings(self.config)
        cache_inputs = build_backtest_feature_cache_inputs(
            namespace = self.name,
            stage = stage,
            state_start_index = stage_start_index,
            spread_segment = spread_segment,
            formation_spread = formation_spread,
            feature_settings = feature_settings,
        )
        cache_key = build_backtest_feature_cache_key(cache_inputs)
        feature_cache_path, feature_manifest_path = build_backtest_feature_cache_paths(
            self.feature_cache_root,
            namespace = self.name,
            stage = stage,
            state_start_index = stage_start_index,
            depth = int(feature_settings["depth"]),
            cache_key = cache_key,
        )

        if self.cache_episode_features:
            try:
                cached_result = load_feature_build_result(feature_cache_path, feature_manifest_path)
                return cached_result, cache_key, feature_cache_path, feature_manifest_path, True
            except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError):
                pass

        feature_result = build_real_stage_feature_tensor(
            spread_segment,
            formation_spread,
            self.config,
        )

        if self.cache_episode_features:
            scaler = fit_zscore_scaler(formation_spread)
            feature_manifest = build_feature_manifest(
                name = f"{self.name}_{stage}_stage_features",
                feature_spec = dict(feature_result.feature_spec),
                scaler_spec = scaler_to_dict(scaler),
                features = feature_result.features,
                source_spread_path = None,
                source_spread_hash = str(cache_inputs["spread_segment_hash"]),
                run_seed = int(_get_config_value(self.config, ["repro", "seed"], 42)),
                git_commit = None,
                deterministic = bool(
                    _get_config_value(self.config, ["repro", "deterministic_algorithms"], False)
                    or _get_config_value(self.config, ["repro", "pytorch_deterministic"], False)
                ),
                library_name = str(feature_result.feature_spec.get("library", feature_settings["library"])),
                library_version = feature_result.feature_spec.get("library_version"),
                device = str(feature_result.feature_spec.get("device", feature_settings["device"])),
            )
            feature_manifest["cache"] = {
                **cache_inputs,
                "cache_key": cache_key,
            }
            feature_manifest["provenance"]["formation_spread_hash"] = str(
                cache_inputs["formation_spread_hash"]
            )
            save_feature_build_result(feature_cache_path, feature_result)
            save_manifest(feature_manifest, feature_manifest_path)

        return (
            feature_result,
            cache_key if self.cache_episode_features else None,
            feature_cache_path if self.cache_episode_features else None,
            feature_manifest_path if self.cache_episode_features else None,
            False,
        )

    # Build one trained SOT stage plan for the current state episode
    def _build_stage_plan(
        self,
        stage: str,
        context: BacktestDayContext,
    ) -> SOTStagePlan:
        state_start_index = int(context.state_start_index)
        plan_start_index = int(context.current_index)
        episode_horizon_cap = int(_get_config_value(self.config, ["sot", "episode_horizon"], 0))
        remaining = int(len(context.trading_window) - 1 - plan_start_index)
        horizon = remaining
        if episode_horizon_cap > 0:
            horizon = min(horizon, episode_horizon_cap)
        stage_window = context.trading_window.iloc[
            plan_start_index : plan_start_index + horizon + 1
        ].reset_index(drop = True)
        initial_spread = (
            float(stage_window.iloc[0]["spread"]) if len(stage_window) > 0 else 0.0
        )
        state_start_date = (
            _serialize_date(context.trading_window.iloc[plan_start_index]["date"])
            if "date" in context.trading_window.columns and plan_start_index < len(context.trading_window)
            else None
        )
        stage_start_index = state_start_index
        output_dir = self.output_root / stage / f"{stage}_start_{plan_start_index:04d}"
        policy_id = f"sot_{stage}_{plan_start_index:04d}"

        if len(stage_window) < 2 or horizon <= 0:
            return SOTStagePlan(
                stage = stage,
                state_start_index = stage_start_index,
                state_start_date = state_start_date,
                initial_spread = initial_spread,
                horizon = max(horizon, 0),
                threshold = self.threshold,
                output_dir = output_dir,
                policy_id = policy_id,
                training_result = None,
                prefix_ends = np.array([], dtype = np.int32),
                scores = np.array([], dtype = np.float64),
                cumulative_squared_scores = np.array([], dtype = np.float64),
                trigger_prefix_index = None,
                trigger_row_index = None,
                trigger_date = None,
                trigger_spread = None,
                plan_start_index = plan_start_index,
            )

        stage_formation_spread = self.formation_spread

        if stage == "entry":
            training_result = self.pretrained_entry_result
        elif stage == "exit":
            training_result = self.pretrained_exit_result
        else:
            raise ValueError(f"Unsupported SOT stage: {stage!r}")
        if training_result is None:
            raise RuntimeError(f"Pretrained {stage} policy is not available.")

        (
            real_stage_result,
            feature_cache_key,
            feature_cache_path,
            feature_manifest_path,
            feature_cache_hit,
        ) = self._load_or_build_stage_feature_result(
            stage = stage,
            stage_start_index = plan_start_index,
            spread_segment = stage_window["spread"].to_numpy(dtype = np.float64),
            formation_spread = stage_formation_spread,
        )
        if real_stage_result.features.size == 0 or real_stage_result.prefix_ends.size == 0:
            return SOTStagePlan(
                stage = stage,
                state_start_index = stage_start_index,
                state_start_date = state_start_date,
                initial_spread = initial_spread,
                horizon = horizon,
                threshold = self.threshold,
                output_dir = output_dir,
                policy_id = policy_id,
                training_result = training_result,
                prefix_ends = np.array([], dtype = np.int32),
                scores = np.array([], dtype = np.float64),
                cumulative_squared_scores = np.array([], dtype = np.float64),
                trigger_prefix_index = None,
                trigger_row_index = None,
                trigger_date = None,
                trigger_spread = None,
                plan_start_index = plan_start_index,
                feature_spec = dict(real_stage_result.feature_spec),
                feature_cache_key = feature_cache_key,
                feature_cache_path = feature_cache_path,
                feature_manifest_path = feature_manifest_path,
                feature_cache_hit = feature_cache_hit,
            )

        scores = np.asarray(
            score_signature_features(training_result.best_policy, real_stage_result.features),
            dtype = np.float64,
        )
        stop_rule = apply_deterministic_stop_rule(scores, self.threshold)

        trigger_row_index = None
        trigger_date = None
        trigger_spread = None
        if stop_rule.stop_index is not None:
            trigger_row_index = int(plan_start_index + real_stage_result.prefix_ends[stop_rule.stop_index])
            trigger_date = (
                _serialize_date(context.trading_window.iloc[trigger_row_index]["date"])
                if "date" in context.trading_window.columns
                else None
            )
            trigger_spread = float(context.trading_window.iloc[trigger_row_index]["spread"])

        return SOTStagePlan(
            stage = stage,
            state_start_index = stage_start_index,
            state_start_date = state_start_date,
            initial_spread = initial_spread,
            horizon = horizon,
            threshold = self.threshold,
            output_dir = output_dir,
            policy_id = policy_id,
            training_result = training_result,
            prefix_ends = np.asarray(real_stage_result.prefix_ends, dtype = np.int32),
            scores = np.asarray(stop_rule.scores, dtype = np.float64),
            cumulative_squared_scores = np.asarray(stop_rule.cumulative_squared_scores, dtype = np.float64),
            trigger_prefix_index = stop_rule.stop_index,
            trigger_row_index = trigger_row_index,
            trigger_date = trigger_date,
            trigger_spread = trigger_spread,
            plan_start_index = plan_start_index,
            feature_spec = dict(real_stage_result.feature_spec),
            feature_cache_key = feature_cache_key,
            feature_cache_path = feature_cache_path,
            feature_manifest_path = feature_manifest_path,
            feature_cache_hit = feature_cache_hit,
        )

    # Convert one stage plan into today's SOT trading decision
    def _build_stage_decision(
        self,
        plan: SOTStagePlan,
        context: BacktestDayContext,
    ) -> StrategyDecision:
        latest_prefix_index = plan.latest_available_prefix_index(context.current_index)
        metadata: dict[str, Any] = {
            "strategy": "sot",
            "stage": plan.stage,
            "stage_start_index": int(plan.state_start_index),
            "stage_start_date": plan.state_start_date,
            "initial_spread": float(plan.initial_spread),
            "remaining_horizon": int(plan.horizon),
            "threshold": float(plan.threshold),
            "mu": float(self.mu),
            "policy_id": plan.policy_id,
            "output_dir": str(plan.output_dir),
            "trigger_row_index": plan.trigger_row_index,
            "trigger_date": plan.trigger_date,
            "feature_cache_hit": bool(plan.feature_cache_hit),
            "feature_cache_key": plan.feature_cache_key,
            "feature_cache_path": None if plan.feature_cache_path is None else str(plan.feature_cache_path),
            "feature_manifest_path": None
            if plan.feature_manifest_path is None
            else str(plan.feature_manifest_path),
        }

        if latest_prefix_index is not None:
            metadata.update(
                {
                    "latest_prefix_index": int(latest_prefix_index),
                    "latest_prefix_end": int(plan.prefix_ends[latest_prefix_index]),
                    "latest_score": float(plan.scores[latest_prefix_index]),
                    "current_q": float(plan.cumulative_squared_scores[latest_prefix_index]),
                    "prefix_scores_available": int(latest_prefix_index + 1),
                }
            )
        else:
            metadata.update(
                {
                    "latest_prefix_index": None,
                    "latest_prefix_end": None,
                    "latest_score": None,
                    "current_q": 0.0,
                    "prefix_scores_available": 0,
                }
            )

        if plan.training_result is None:
            return StrategyDecision.hold(metadata)

        if plan.trigger_row_index is None or context.current_index < plan.trigger_row_index:
            return StrategyDecision.hold(metadata)

        metadata["trigger_spread"] = plan.trigger_spread
        if plan.stage == "entry":
            return StrategyDecision.enter_long_spread(metadata)
        if plan.stage == "exit":
            return StrategyDecision.exit_long_spread(metadata)
        raise ValueError(f"Unsupported SOT stage: {plan.stage!r}")
