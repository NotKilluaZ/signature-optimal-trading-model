from __future__ import annotations
import json
from pathlib import Path
from uuid import uuid4
import numpy as np
from src.sigstop.train.trainer import (
    LinearStoppingTrainingConfig,
    compute_policy_loss_and_gradients,
    load_training_checkpoint,
    train_linear_stopping_policy,
)
from src.sigstop.stopping.policy import LinearStoppingPolicy


def _finite_difference_gradient(
    func,
    x: np.ndarray,
    *,
    eps: float = 1.0e-6,
) -> np.ndarray:
    grad = np.empty_like(x, dtype = np.float64)
    for index in range(x.size):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[index] += eps
        x_minus[index] -= eps
        grad[index] = (func(x_plus) - func(x_minus)) / (2.0 * eps)
    return grad


def test_policy_gradient_matches_finite_difference_on_weights_and_bias() -> None:
    features = np.array(
        [
            [[1.0, 0.5], [0.7, -0.2], [0.4, 1.2]],
            [[0.8, -0.3], [1.1, 0.6], [0.2, 0.9]],
        ],
        dtype = np.float64,
    )
    payoffs = np.array(
        [
            [0.0, -0.5, -1.0, -1.5],
            [0.0, -0.25, -0.75, -1.1],
        ],
        dtype = np.float64,
    )
    policy = LinearStoppingPolicy(
        weights = np.array([0.11, -0.07], dtype = np.float64),
        bias = 0.03,
        use_bias = True,
    )

    gradient_result = compute_policy_loss_and_gradients(
        policy,
        features,
        payoffs,
        threshold = 0.05,
        mu = 20.0,
        weight_decay = 0.0,
    )

    def objective_weight(local_weights: np.ndarray) -> float:
        local_policy = LinearStoppingPolicy(
            weights = local_weights,
            bias = policy.bias,
            use_bias = True,
        )
        return compute_policy_loss_and_gradients(
            local_policy,
            features,
            payoffs,
            threshold = 0.05,
            mu = 20.0,
            weight_decay = 0.0,
        ).total_loss

    def objective_bias(local_bias_array: np.ndarray) -> float:
        local_policy = LinearStoppingPolicy(
            weights = policy.weights,
            bias = float(local_bias_array[0]),
            use_bias = True,
        )
        return compute_policy_loss_and_gradients(
            local_policy,
            features,
            payoffs,
            threshold = 0.05,
            mu = 20.0,
            weight_decay = 0.0,
        ).total_loss

    numerical_weight_grad = _finite_difference_gradient(objective_weight, policy.weights)
    numerical_bias_grad = _finite_difference_gradient(
        objective_bias,
        np.array([policy.bias], dtype = np.float64),
    )[0]

    np.testing.assert_allclose(gradient_result.weight_grad, numerical_weight_grad, atol = 1.0e-5, rtol = 1.0e-4)
    np.testing.assert_allclose(gradient_result.bias_grad, numerical_bias_grad, atol = 1.0e-5, rtol = 1.0e-4)


def test_training_loop_saves_artifacts_and_checkpoint_metadata() -> None:
    features = np.ones((12, 4, 3), dtype = np.float32)
    payoffs = np.tile(np.array([0.0, -1.0, -2.0, -3.0, -4.0], dtype = np.float64), (12, 1))

    config = LinearStoppingTrainingConfig(
        threshold = 0.05,
        mu = 20.0,
        batch_size = 4,
        max_epochs = 12,
        validation_fraction = 0.25,
        shuffle = True,
        seed = 7,
        learning_rate = 1.0e-2,
        early_stopping_enabled = True,
        early_stopping_patience = 4,
        include_bias = False,
        init = "small_normal",
        init_weight_scale = 1.0e-2,
        dtype = "float32",
        device = "cpu",
    )

    output_dir = Path("runs") / "pytest_artifacts" / f"entry_train_{uuid4().hex}"
    result = train_linear_stopping_policy(
        features,
        payoffs,
        config,
        output_dir = output_dir,
        stage = "entry",
        run_id = "test_run",
        extra_manifest_data = {"note": "trainer_test"},
    )

    assert result.history.epochs_completed >= 1
    assert result.artifacts is not None
    assert result.artifacts.checkpoint_last_path.exists()
    assert result.artifacts.policy_best_path.exists()
    assert result.artifacts.optimizer_last_path.exists()
    assert result.artifacts.optimizer_best_path.exists()
    assert result.artifacts.manifest_path.exists()
    assert result.history.best_monitored_loss is not None
    assert result.history.train_loss[0] >= result.history.best_monitored_loss

    manifest = json.loads(result.artifacts.manifest_path.read_text(encoding = "utf-8"))
    assert manifest["stage"] == "entry"
    assert manifest["run_id"] == "test_run"
    assert manifest["provenance"]["note"] == "trainer_test"

    checkpoint = load_training_checkpoint(result.artifacts.checkpoint_last_path)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert int(checkpoint["epoch"]) >= 0
