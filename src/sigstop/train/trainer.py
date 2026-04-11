from __future__ import annotations
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any
import numpy as np
import torch
from src.sigstop.features.manifest import compute_file_sha256, safe_package_version
from src.sigstop.paths import RUNS_DIR
from src.sigstop.stopping.policy import LinearStoppingPolicy, validate_signature_feature_tensor
from src.sigstop.stopping.torch_loss import build_torch_smoothed_loss_result
from src.sigstop.stopping.torch_policy import (
    build_torch_policy_from_linear_policy,
    initialize_torch_policy_from_feature_tensor,
    resolve_torch_device,
    resolve_torch_dtype
)


@dataclass(frozen = True)
class LinearStoppingTrainingConfig:
    threshold: float
    mu: float
    batch_size: int = 128
    max_epochs: int = 200
    validation_fraction: float = 0.2
    shuffle: bool = True
    seed: int = 42
    optimizer_name: str = "adam"
    learning_rate: float = 1.0e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1.0e-8
    weight_decay: float = 0.0
    grad_clip_enabled: bool = True
    grad_clip_max_norm: float = 5.0
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 1.0e-4
    include_bias: bool = False
    scalar_term_in_signatures: bool = True
    init: str = "small_normal"
    init_weight_scale: float = 1.0e-2
    dtype: str = "float32"
    device: str = "cuda_if_available"
    torch_manual_seed: bool = True
    deterministic_algorithms: bool = False

    # Convert config dataclass into dictionary
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyGradientResult:
    total_loss: float
    base_loss: float
    reg_loss: float
    weight_grad: np.ndarray
    bias_grad: float
    grad_norm: float


@dataclass
class TrainingHistory:
    epoch: list[int] = field(default_factory = list)
    train_loss: list[float] = field(default_factory = list)
    train_objective: list[float] = field(default_factory = list)
    val_loss: list[float | None] = field(default_factory = list)
    monitored_loss: list[float] = field(default_factory = list)
    average_grad_norm: list[float] = field(default_factory = list)
    best_epoch: int | None = None
    best_monitored_loss: float | None = None
    stopped_early: bool = False
    epochs_completed: int = 0

    # Convert training history dataclass into dictionary
    def to_dict(self) -> dict[str, Any]:
        return {
            "epoch": list(self.epoch),
            "train_loss": list(self.train_loss),
            "train_objective": list(self.train_objective),
            "val_loss": list(self.val_loss),
            "monitored_loss": list(self.monitored_loss),
            "average_grad_norm": list(self.average_grad_norm),
            "best_epoch": self.best_epoch,
            "best_monitored_loss": self.best_monitored_loss,
            "stopped_early": self.stopped_early,
            "epochs_completed": self.epochs_completed,
        }


@dataclass(frozen = True)
class TrainingArtifacts:
    output_dir: Path
    checkpoint_last_path: Path
    checkpoint_best_path: Path
    policy_best_path: Path
    optimizer_last_path: Path
    optimizer_best_path: Path
    history_path: Path
    manifest_path: Path


@dataclass
class StoppingTrainingData:
    features: np.ndarray
    payoffs: np.ndarray
    prefix_ends: np.ndarray
    feature_spec: dict[str, Any]
    scaler_spec: dict[str, Any]
    metadata: dict[str, Any]


@dataclass
class TrainingResult:
    policy: LinearStoppingPolicy
    best_policy: LinearStoppingPolicy
    optimizer_state: dict[str, Any]
    history: TrainingHistory
    config: LinearStoppingTrainingConfig
    artifacts: TrainingArtifacts | None = None
    train_features_shape: tuple[int, ...] | None = None
    train_payoffs_shape: tuple[int, ...] | None = None
    val_features_shape: tuple[int, ...] | None = None
    val_payoffs_shape: tuple[int, ...] | None = None


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if isinstance(key, int):
            if not isinstance(current, list) or key >= len(current):
                return default
        elif not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Build training config dataclass from the YAML config dictionary
def build_training_config_from_dict(config: dict[str, Any]) -> LinearStoppingTrainingConfig:
    return LinearStoppingTrainingConfig(
        threshold = float(_get_config_value(config, ["stopping", "deterministic_threshold", "k"], 0.05)),
        mu = float(_get_config_value(config, ["stopping", "deterministic_threshold", "mu"], 20.0)),
        batch_size = int(_get_config_value(config, ["training", "batch_size"], 128)),
        max_epochs = int(_get_config_value(config, ["training", "schedule", "max_epochs"], 200)),
        validation_fraction = float(_get_config_value(config, ["training", "validation", "fraction"], 0.2)),
        shuffle = bool(_get_config_value(config, ["training", "validation", "shuffle"], True)),
        seed = int(_get_config_value(config, ["repro", "seed"], 42)),
        optimizer_name = str(_get_config_value(config, ["training", "optimizer", "name"], "adam")).lower(),
        learning_rate = float(_get_config_value(config, ["training", "optimizer", "lr"], 1.0e-3)),
        beta1 = float(_get_config_value(config, ["training", "optimizer", "betas", 0], 0.9)),
        beta2 = float(_get_config_value(config, ["training", "optimizer", "betas", 1], 0.999)),
        eps = float(_get_config_value(config, ["training", "optimizer", "eps"], 1.0e-8)),
        weight_decay = float(_get_config_value(config, ["training", "optimizer", "weight_decay"], 0.0)),
        grad_clip_enabled = bool(_get_config_value(config, ["training", "regularization", "grad_clip", "enabled"], True)),
        grad_clip_max_norm = float(_get_config_value(config, ["training", "regularization", "grad_clip", "max_norm"], 5.0)),
        early_stopping_enabled = bool(_get_config_value(config, ["training", "schedule", "early_stopping", "enabled"], True)),
        early_stopping_patience = int(_get_config_value(config, ["training", "schedule", "early_stopping", "patience"], 20)),
        early_stopping_min_delta = float(_get_config_value(config, ["training", "schedule", "early_stopping", "min_delta"], 1.0e-4)),
        include_bias = bool(_get_config_value(config, ["stopping", "policy", "include_bias"], False)),
        scalar_term_in_signatures = bool(_get_config_value(config, ["stopping", "policy", "scalar_term_in_signatures"], True)),
        init = str(_get_config_value(config, ["stopping", "policy", "init"], "small_normal")),
        init_weight_scale = float(_get_config_value(config, ["stopping", "policy", "init_weight_scale"], 1.0e-2)),
        dtype = str(_get_config_value(config, ["features", "signature", "dtype"], "float32")),
        device = str(_get_config_value(config, ["training", "device"], "cuda_if_available")),
        torch_manual_seed = bool(_get_config_value(config, ["repro", "torch_manual_seed"], True)),
        deterministic_algorithms = bool(
            _get_config_value(
                config,
                ["repro", "deterministic_algorithms"],
                _get_config_value(config, ["repro", "pytorch_deterministic"], False),
            )
        ),
    )


# Create a timestamped default run id
def build_default_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# Build the default output directory for one training stage
def build_default_stage_output_dir(stage: str, *, run_id: str | None = None) -> tuple[str, Path]:
    resolved_run_id = run_id or build_default_run_id(prefix = "stopping")
    return resolved_run_id, RUNS_DIR / resolved_run_id / "stopping_policy" / stage


# Validate and coerce payoff arrays into batched 2D form
def _coerce_batched_payoff_process(
    payoffs: np.ndarray | list[float] | list[list[float]],
) -> np.ndarray:
    x = np.asarray(payoffs, dtype = float)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"Payoff process must have shape (L,) or (B, L). Got shape: {x.shape}")
    if x.shape[1] < 2:
        raise ValueError("Payoff process must contain at least 2 observations.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Payoff process contains NaN or infinite value(s).")
    return x


# Validate and align feature and payoff tensors for training
def validate_training_inputs(
    features: np.ndarray | list[list[float]] | list[list[list[float]]],
    payoffs: np.ndarray | list[float] | list[list[float]],
) -> tuple[np.ndarray, np.ndarray]:
    x, _ = validate_signature_feature_tensor(features)
    y = _coerce_batched_payoff_process(payoffs)
    if y.shape[1] != x.shape[1] + 1:
        raise ValueError(
            "Payoffs must have exactly one more time step than feature prefixes. "
            f"Got features shape={x.shape} and payoffs shape={y.shape}."
        )
    if x.shape[0] == y.shape[0]:
        return x, y
    if x.shape[0] == 1:
        return np.broadcast_to(x, (y.shape[0], x.shape[1], x.shape[2])), y
    if y.shape[0] == 1:
        return x, np.broadcast_to(y, (x.shape[0], y.shape[1]))
    raise ValueError(
        "Features and payoffs must have the same batch size or be broadcastable from one path. "
        f"Got features batch={x.shape[0]} and payoffs batch={y.shape[0]}."
    )


# Split simulated paths into train and validation subsets
def split_train_validation_paths(
    features: np.ndarray,
    payoffs: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    if validation_fraction < 0.0 or validation_fraction >= 1.0:
        raise ValueError(f"validation_fraction must be in [0, 1). Got: {validation_fraction}")
    n_paths = features.shape[0]
    if validation_fraction == 0.0 or n_paths == 1:
        return features, payoffs, None, None
    n_val = min(max(int(np.floor(n_paths * validation_fraction)), 1), n_paths - 1)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_paths)
    val_indices = permutation[:n_val]
    train_indices = permutation[n_val:]
    return features[train_indices], payoffs[train_indices], features[val_indices], payoffs[val_indices]


# Set torch reproducibility flags for training
def set_torch_reproducibility(config: LinearStoppingTrainingConfig) -> None:
    if config.torch_manual_seed:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic_algorithms)


# Convert numpy arrays into torch tensors on the target device
def to_torch_batch(
    values: np.ndarray,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(values, dtype = dtype, device = device)


# Evaluate the smoothed stopping loss for a serializable policy
def evaluate_policy_loss(
    policy: LinearStoppingPolicy,
    features: np.ndarray,
    payoffs: np.ndarray,
    *,
    threshold: float,
    mu: float,
    weight_decay: float = 0.0,
    include_regularization: bool = False,
    device: str | torch.device = "cpu",
) -> tuple[float, float]:
    module = build_torch_policy_from_linear_policy(policy, device = device)
    module.eval()
    resolved_dtype = module.weights.dtype
    x = to_torch_batch(features, dtype = resolved_dtype, device = module.weights.device)
    y = to_torch_batch(payoffs, dtype = resolved_dtype, device = module.weights.device)

    with torch.no_grad():
        result = build_torch_smoothed_loss_result(module(x), y, threshold = threshold, mu = mu)
        reg_loss = 0.0
        if include_regularization and weight_decay > 0.0:
            reg_loss = 0.5 * float(weight_decay) * float(torch.sum(torch.square(module.weights)).item())
        return float(result.loss.item()), float(reg_loss)


# Compute the smoothed stopping loss and torch autograd gradients
def compute_policy_loss_and_gradients(
    policy: LinearStoppingPolicy,
    features: np.ndarray | list[list[float]] | list[list[list[float]]],
    payoffs: np.ndarray | list[float] | list[list[float]],
    *,
    threshold: float,
    mu: float,
    weight_decay: float = 0.0,
    device: str | torch.device = "cpu",
) -> PolicyGradientResult:
    x_np, y_np = validate_training_inputs(features, payoffs)
    module = build_torch_policy_from_linear_policy(policy, device = device)
    module.train()
    resolved_dtype = module.weights.dtype
    x = to_torch_batch(x_np, dtype = resolved_dtype, device = module.weights.device)
    y = to_torch_batch(y_np, dtype = resolved_dtype, device = module.weights.device)

    module.zero_grad(set_to_none = True)
    result = build_torch_smoothed_loss_result(module(x), y, threshold = threshold, mu = mu)
    reg_loss = torch.zeros((), dtype = resolved_dtype, device = module.weights.device)
    if weight_decay > 0.0:
        reg_loss = 0.5 * float(weight_decay) * torch.sum(torch.square(module.weights))
    total_loss = result.loss + reg_loss
    total_loss.backward()

    weight_grad = module.weights.grad.detach().cpu().numpy().copy()
    bias_grad = 0.0 if module.bias is None or module.bias.grad is None else float(module.bias.grad.detach().cpu().item())
    grad_norm_sq = float(np.sum(np.square(weight_grad)) + bias_grad ** 2)

    return PolicyGradientResult(
        total_loss = float(total_loss.detach().cpu().item()),
        base_loss = float(result.loss.detach().cpu().item()),
        reg_loss = float(reg_loss.detach().cpu().item()),
        weight_grad = weight_grad,
        bias_grad = bias_grad,
        grad_norm = float(np.sqrt(grad_norm_sq)),
    )


# Build minibatch index slices for one training epoch
def iterate_minibatch_indices(
    n_paths: int,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    epoch: int,
) -> list[np.ndarray]:
    indices = np.arange(n_paths, dtype = np.int32)
    if shuffle:
        np.random.default_rng(seed + epoch).shuffle(indices)
    return [indices[start : start + batch_size] for start in range(0, n_paths, batch_size)]


# Build standard artifact paths for a training run
def build_training_artifacts(output_dir: str | Path) -> TrainingArtifacts:
    resolved = Path(output_dir)
    resolved.mkdir(parents = True, exist_ok = True)
    return TrainingArtifacts(
        output_dir = resolved,
        checkpoint_last_path = resolved / "checkpoint_last.pt",
        checkpoint_best_path = resolved / "checkpoint_best.pt",
        policy_best_path = resolved / "policy_best.pt",
        optimizer_last_path = resolved / "optimizer_last.pt",
        optimizer_best_path = resolved / "optimizer_best.pt",
        history_path = resolved / "history.json",
        manifest_path = resolved / "manifest.json",
    )


# Save a JSON payload to disk
def save_json(payload: dict[str, Any], path: str | Path) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents = True, exist_ok = True)
    with open(resolved, "w", encoding = "utf-8") as handle:
        json.dump(payload, handle, indent = 2)


# Load a JSON payload from disk
def load_json(path: str | Path) -> dict[str, Any]:
    with open(Path(path), "r", encoding = "utf-8") as handle:
        return json.load(handle)


# Save a policy snapshot to disk
def save_policy(path: str | Path, policy: LinearStoppingPolicy) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents = True, exist_ok = True)
    torch.save(policy.to_dict(), resolved)


# Load a policy snapshot from disk
def load_policy(path: str | Path) -> LinearStoppingPolicy:
    payload = torch.load(Path(path), map_location = "cpu")
    return LinearStoppingPolicy(
        weights = np.asarray(payload["weights"]),
        bias = float(payload["bias"]),
        use_bias = bool(payload["use_bias"]),
        scalar_term_in_signatures = bool(payload["scalar_term_in_signatures"]),
        init = str(payload["init"]),
    )


# Save a standalone optimizer state dictionary to disk
def save_optimizer_state(path: str | Path, optimizer: torch.optim.Optimizer) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents = True, exist_ok = True)
    torch.save(optimizer.state_dict(), resolved)


# Move optimizer state tensors onto the target device
def move_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


# Save a full torch training checkpoint with optimizer state
def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_epoch: int | None,
    best_monitored_loss: float | None,
    config: LinearStoppingTrainingConfig,
) -> None:
    resolved = Path(path)
    resolved.parent.mkdir(parents = True, exist_ok = True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "best_epoch": -1 if best_epoch is None else int(best_epoch),
            "best_monitored_loss": None if best_monitored_loss is None else float(best_monitored_loss),
            "config": config.to_dict(),
        },
        resolved,
    )


# Load a full torch training checkpoint with optimizer state
def load_training_checkpoint(path: str | Path) -> dict[str, Any]:
    return torch.load(Path(path), map_location = "cpu")


# Build a reproducibility manifest for one training run
def build_training_manifest(
    *,
    stage: str,
    run_id: str | None,
    config: LinearStoppingTrainingConfig,
    history: TrainingHistory,
    best_policy: LinearStoppingPolicy,
    train_features: np.ndarray,
    train_payoffs: np.ndarray,
    val_features: np.ndarray | None,
    val_payoffs: np.ndarray | None,
    artifacts: TrainingArtifacts,
    resolved_device: torch.device,
    extra_manifest_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_hashes = {
        key: compute_file_sha256(path) if path.exists() else None
        for key, path in {
            "checkpoint_last": artifacts.checkpoint_last_path,
            "checkpoint_best": artifacts.checkpoint_best_path,
            "policy_best": artifacts.policy_best_path,
            "optimizer_last": artifacts.optimizer_last_path,
            "optimizer_best": artifacts.optimizer_best_path,
            "history": artifacts.history_path,
        }.items()
    }
    return {
        "stage": stage,
        "run_id": run_id,
        "training_config": config.to_dict(),
        "history_summary": {
            "epochs_completed": history.epochs_completed,
            "best_epoch": history.best_epoch,
            "best_monitored_loss": history.best_monitored_loss,
            "stopped_early": history.stopped_early,
            "final_train_loss": None if not history.train_loss else history.train_loss[-1],
            "final_val_loss": None if not history.val_loss else history.val_loss[-1],
        },
        "best_policy": best_policy.to_dict(),
        "dataset": {
            "train_features_shape": list(train_features.shape),
            "train_payoffs_shape": list(train_payoffs.shape),
            "val_features_shape": None if val_features is None else list(val_features.shape),
            "val_payoffs_shape": None if val_payoffs is None else list(val_payoffs.shape),
        },
        "artifacts": {
            "output_dir": str(artifacts.output_dir),
            "checkpoint_last_path": str(artifacts.checkpoint_last_path),
            "checkpoint_best_path": str(artifacts.checkpoint_best_path),
            "policy_best_path": str(artifacts.policy_best_path),
            "optimizer_last_path": str(artifacts.optimizer_last_path),
            "optimizer_best_path": str(artifacts.optimizer_best_path),
            "history_path": str(artifacts.history_path),
            "hashes": artifact_hashes,
        },
        "repro": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "os_name": os.name,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(resolved_device),
        },
        "provenance": extra_manifest_data or {},
    }


# Train a linear stopping policy with torch Adam, batching, and early stopping
def train_linear_stopping_policy(
    features: np.ndarray | list[list[float]] | list[list[list[float]]],
    payoffs: np.ndarray | list[float] | list[list[float]],
    config: LinearStoppingTrainingConfig,
    *,
    output_dir: str | Path | None = None,
    stage: str = "entry",
    run_id: str | None = None,
    initial_policy: LinearStoppingPolicy | None = None,
    resume_from: str | Path | None = None,
    extra_manifest_data: dict[str, Any] | None = None,
) -> TrainingResult:
    if config.optimizer_name != "adam":
        raise ValueError(f"Unsupported optimizer {config.optimizer_name!r}. Expected 'adam'.")

    set_torch_reproducibility(config)
    resolved_device = resolve_torch_device(config.device)
    resolved_dtype = resolve_torch_dtype(config.dtype)

    all_features, all_payoffs = validate_training_inputs(features, payoffs)
    train_features, train_payoffs, val_features, val_payoffs = split_train_validation_paths(
        all_features,
        all_payoffs,
        validation_fraction = config.validation_fraction,
        seed = config.seed,
    )

    train_x = to_torch_batch(train_features, dtype = resolved_dtype, device = resolved_device)
    train_y = to_torch_batch(train_payoffs, dtype = resolved_dtype, device = resolved_device)
    val_x = None if val_features is None else to_torch_batch(val_features, dtype = resolved_dtype, device = resolved_device)
    val_y = None if val_payoffs is None else to_torch_batch(val_payoffs, dtype = resolved_dtype, device = resolved_device)

    artifacts = None if output_dir is None else build_training_artifacts(output_dir)
    if resume_from is not None and initial_policy is not None:
        raise ValueError("Provide either resume_from or initial_policy, not both.")

    if resume_from is not None:
        checkpoint = load_training_checkpoint(resume_from)
        policy_module = initialize_torch_policy_from_feature_tensor(
            train_features,
            include_bias = bool(checkpoint["config"]["include_bias"]),
            init = str(checkpoint["config"]["init"]),
            weight_scale = float(checkpoint["config"]["init_weight_scale"]),
            seed = config.seed,
            dtype = config.dtype,
            device = resolved_device,
            scalar_term_in_signatures = bool(checkpoint["config"]["scalar_term_in_signatures"]),
        )
        policy_module.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.Adam(
            policy_module.parameters(),
            lr = config.learning_rate,
            betas = (config.beta1, config.beta2),
            eps = config.eps,
            weight_decay = config.weight_decay,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        move_optimizer_state_to_device(optimizer, resolved_device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_epoch = None if int(checkpoint["best_epoch"]) < 0 else int(checkpoint["best_epoch"])
        best_monitored_loss = checkpoint["best_monitored_loss"]
    else:
        policy_module = build_torch_policy_from_linear_policy(initial_policy, device = resolved_device) if initial_policy is not None else initialize_torch_policy_from_feature_tensor(
            train_features,
            include_bias = config.include_bias,
            init = config.init,
            weight_scale = config.init_weight_scale,
            seed = config.seed,
            dtype = config.dtype,
            device = resolved_device,
            scalar_term_in_signatures = config.scalar_term_in_signatures,
        )
        optimizer = torch.optim.Adam(
            policy_module.parameters(),
            lr = config.learning_rate,
            betas = (config.beta1, config.beta2),
            eps = config.eps,
            weight_decay = config.weight_decay,
        )
        start_epoch = 0
        best_epoch = None
        best_monitored_loss = None

    history = TrainingHistory()
    best_policy = policy_module.to_linear_policy()

    def evaluate_current(module: torch.nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
        module.eval()
        with torch.no_grad():
            return float(
                build_torch_smoothed_loss_result(
                    module(x),
                    y,
                    threshold = config.threshold,
                    mu = config.mu,
                ).loss.detach().cpu().item()
            )

    if best_monitored_loss is None:
        best_monitored_loss = evaluate_current(policy_module, train_x, train_y)
        if val_x is not None and val_y is not None:
            best_monitored_loss = evaluate_current(policy_module, val_x, val_y)

    epochs_without_improvement = 0

    for epoch in range(start_epoch, config.max_epochs):
        policy_module.train()
        batch_grad_norms: list[float] = []
        for batch_index in iterate_minibatch_indices(
            train_features.shape[0],
            batch_size = config.batch_size,
            shuffle = config.shuffle,
            seed = config.seed,
            epoch = epoch,
        ):
            optimizer.zero_grad(set_to_none = True)
            batch_loss = build_torch_smoothed_loss_result(
                policy_module(train_x[batch_index]),
                train_y[batch_index],
                threshold = config.threshold,
                mu = config.mu,
            ).loss
            batch_loss.backward()

            if config.grad_clip_enabled:
                grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        policy_module.parameters(),
                        max_norm = config.grad_clip_max_norm,
                    ).detach().cpu().item()
                )
            else:
                grad_norm_sq = 0.0
                for parameter in policy_module.parameters():
                    if parameter.grad is not None:
                        grad_norm_sq += float(torch.sum(parameter.grad.detach() ** 2).cpu().item())
                grad_norm = float(np.sqrt(grad_norm_sq))

            optimizer.step()
            batch_grad_norms.append(grad_norm)

        train_loss = evaluate_current(policy_module, train_x, train_y)
        reg_loss = 0.0 if config.weight_decay == 0.0 else 0.5 * config.weight_decay * float(torch.sum(policy_module.weights.detach() ** 2).cpu().item())
        train_objective = train_loss + reg_loss
        val_loss = None if val_x is None or val_y is None else evaluate_current(policy_module, val_x, val_y)
        monitored_loss = train_loss if val_loss is None else val_loss

        history.epoch.append(epoch)
        history.train_loss.append(train_loss)
        history.train_objective.append(train_objective)
        history.val_loss.append(val_loss)
        history.monitored_loss.append(monitored_loss)
        history.average_grad_norm.append(float(np.mean(batch_grad_norms)))
        history.epochs_completed = epoch + 1

        if monitored_loss < float(best_monitored_loss) - config.early_stopping_min_delta:
            best_monitored_loss = float(monitored_loss)
            best_epoch = epoch
            best_policy = policy_module.to_linear_policy()
            history.best_epoch = best_epoch
            history.best_monitored_loss = best_monitored_loss
            epochs_without_improvement = 0

            if artifacts is not None:
                save_policy(artifacts.policy_best_path, best_policy)
                save_optimizer_state(artifacts.optimizer_best_path, optimizer)
                save_training_checkpoint(
                    artifacts.checkpoint_best_path,
                    model = policy_module,
                    optimizer = optimizer,
                    epoch = epoch,
                    best_epoch = best_epoch,
                    best_monitored_loss = best_monitored_loss,
                    config = config,
                )
        else:
            epochs_without_improvement += 1
            history.best_epoch = best_epoch
            history.best_monitored_loss = best_monitored_loss

        if artifacts is not None:
            save_optimizer_state(artifacts.optimizer_last_path, optimizer)
            save_training_checkpoint(
                artifacts.checkpoint_last_path,
                model = policy_module,
                optimizer = optimizer,
                epoch = epoch,
                best_epoch = best_epoch,
                best_monitored_loss = best_monitored_loss,
                config = config,
            )
            save_json(history.to_dict(), artifacts.history_path)

        if config.early_stopping_enabled and epochs_without_improvement >= config.early_stopping_patience:
            history.stopped_early = True
            break

    if artifacts is not None:
        if not artifacts.policy_best_path.exists():
            save_policy(artifacts.policy_best_path, best_policy)
        save_json(
            build_training_manifest(
                stage = stage,
                run_id = run_id,
                config = config,
                history = history,
                best_policy = best_policy,
                train_features = train_features,
                train_payoffs = train_payoffs,
                val_features = val_features,
                val_payoffs = val_payoffs,
                artifacts = artifacts,
                resolved_device = resolved_device,
                extra_manifest_data = extra_manifest_data,
            ),
            artifacts.manifest_path,
        )

    return TrainingResult(
        policy = policy_module.to_linear_policy(),
        best_policy = best_policy,
        optimizer_state = optimizer.state_dict(),
        history = history,
        config = config,
        artifacts = artifacts,
        train_features_shape = tuple(train_features.shape),
        train_payoffs_shape = tuple(train_payoffs.shape),
        val_features_shape = None if val_features is None else tuple(val_features.shape),
        val_payoffs_shape = None if val_payoffs is None else tuple(val_payoffs.shape),
    )
