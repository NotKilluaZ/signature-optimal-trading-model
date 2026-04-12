from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np

SUPPORTED_POLICY_INITS = {
    "small_normal",
    "zeros",
}

# Resolve and validate the policy data type
def _resolve_dtype(dtype: str | np.dtype | type[np.floating]) -> np.dtype:
    resolved = np.dtype(dtype)
    if resolved not in {np.dtype(np.float32), np.dtype(np.float64)}:
        raise ValueError(f"Unsupported policy dtype. Got: {resolved}")
    return resolved

# Validate policy weight vector
def validate_weight_vector(weights: np.ndarray | list[float]) -> np.ndarray:
    x = np.asarray(weights)

    if x.ndim != 1:
        raise ValueError(f"Policy weights must be a 1D array. Got shape: {x.shape}")
    if x.size == 0:
        raise ValueError("Policy weights must contain at least one coefficient.")
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    if not np.all(np.isfinite(x)):
        raise ValueError("Policy weights contain NaN or infinite value(s).")

    return x

# Validate signature feature tensor
def validate_signature_feature_tensor(
    features: np.ndarray | list[list[float]] | list[list[list[float]]],
) -> tuple[np.ndarray, bool]:
    x = np.asarray(features)
    if not np.issubdtype(x.dtype, np.floating):
        x = x.astype(np.float64)
    squeezed = False

    if x.ndim == 2:
        x = x[None, :, :]
        squeezed = True
    elif x.ndim != 3:
        raise ValueError(
            "Signature features must have shape (P, D) or (B, P, D). "
            f"Got shape: {x.shape}"
        )

    if x.shape[1] == 0:
        raise ValueError("Feature tensor must contain at least one prefix row.")
    if x.shape[2] == 0:
        raise ValueError("Feature tensor must contain at least one feature channel.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Feature tensor contains NaN or infinite value(s).")

    return x, squeezed

# Store linear stopping policy parameters
@dataclass
class LinearStoppingPolicy:
    weights: np.ndarray
    bias: float = 0.0
    use_bias: bool = False
    scalar_term_in_signatures: bool = True
    init: str = "manual"

    # Validate policy parameters after initialization
    def __post_init__(self) -> None:
        resolved_weights = validate_weight_vector(self.weights)
        resolved_bias = float(self.bias)

        if not np.isfinite(resolved_bias):
            raise ValueError(f"Policy bias must be finite. Got: {self.bias}")

        self.weights = resolved_weights
        self.bias = resolved_bias
        self.use_bias = bool(self.use_bias)
        self.scalar_term_in_signatures = bool(self.scalar_term_in_signatures)
        self.init = str(self.init)

    # Get number of input features expected by the policy
    @property
    def input_dim(self) -> int:
        return int(self.weights.shape[0])

    # Convert policy parameters into dictionary form
    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "use_bias": bool(self.use_bias),
            "scalar_term_in_signatures": bool(self.scalar_term_in_signatures),
            "input_dim": self.input_dim,
            "init": self.init,
            "dtype": str(self.weights.dtype),
        }

# Initialize linear stopping policy
def initialize_linear_policy(
    input_dim: int,
    *,
    include_bias: bool = False,
    init: str = "small_normal",
    weight_scale: float = 1.0e-2,
    seed: int | None = None,
    dtype: str | np.dtype | type[np.floating] = np.float32,
    scalar_term_in_signatures: bool = True,
) -> LinearStoppingPolicy:
    if input_dim <= 0:
        raise ValueError(f"input_dim must be positive. Got: {input_dim}")
    if init not in SUPPORTED_POLICY_INITS:
        raise ValueError(
            f"Unsupported policy init {init!r}. "
            f"Expected one of: {sorted(SUPPORTED_POLICY_INITS)}"
        )

    resolved_dtype = _resolve_dtype(dtype)
    resolved_scale = float(weight_scale)

    if not np.isfinite(resolved_scale):
        raise ValueError(f"weight_scale must be finite. Got: {weight_scale}")
    if resolved_scale < 0.0:
        raise ValueError(f"weight_scale must be non-negative. Got: {weight_scale}")

    rng = np.random.default_rng(seed)

    if init == "small_normal":
        weights = rng.normal(loc = 0.0, scale = resolved_scale, size = input_dim)
    else:
        weights = np.zeros(input_dim, dtype = np.float64)

    return LinearStoppingPolicy(
        weights = np.asarray(weights, dtype = resolved_dtype),
        bias = 0.0,
        use_bias = include_bias,
        scalar_term_in_signatures = scalar_term_in_signatures,
        init = init,
    )

# Initialize linear policy from feature tensor shape
def initialize_policy_from_feature_tensor(
    features: np.ndarray | list[list[float]] | list[list[list[float]]],
    *,
    include_bias: bool = False,
    init: str = "small_normal",
    weight_scale: float = 1.0e-2,
    seed: int | None = None,
    dtype: str | np.dtype | type[np.floating] = np.float32,
    scalar_term_in_signatures: bool = True,
) -> LinearStoppingPolicy:
    x, _ = validate_signature_feature_tensor(features)
    return initialize_linear_policy(
        input_dim = int(x.shape[-1]),
        include_bias = include_bias,
        init = init,
        weight_scale = weight_scale,
        seed = seed,
        dtype = dtype,
        scalar_term_in_signatures = scalar_term_in_signatures,
    )

# Compute policy scores from signature features
def score_signature_features(
    policy: LinearStoppingPolicy,
    features: np.ndarray | list[list[float]] | list[list[list[float]]],
) -> np.ndarray:
    x, squeezed = validate_signature_feature_tensor(features)

    if x.shape[-1] != policy.input_dim:
        raise ValueError(
            "Policy input dimension does not match feature tensor width. "
            f"Policy input_dim={policy.input_dim}, feature_dim={x.shape[-1]}."
        )

    scores = np.tensordot(x, policy.weights, axes = ([-1], [0]))
    if policy.use_bias:
        scores = scores + policy.bias

    if not np.all(np.isfinite(scores)):
        raise ValueError("Policy scores contain NaN or infinite value(s).")

    return scores[0] if squeezed else scores
