from __future__ import annotations
import numpy as np
import torch
from torch import nn
from src.sigstop.stopping.policy import LinearStoppingPolicy

SUPPORTED_TORCH_POLICY_INITS = {
    "small_normal",
    "zeros",
}

# Resolve and validate the torch device setting
def resolve_torch_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device) == "cuda_if_available":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

# Resolve and validate the torch data type
def resolve_torch_dtype(
    dtype: str | np.dtype | torch.dtype | type[np.floating] = torch.float32,
) -> torch.dtype:
    if dtype in {torch.float32, "float32", np.float32, np.dtype(np.float32)}:
        return torch.float32
    if dtype in {torch.float64, "float64", np.float64, np.dtype(np.float64)}:
        return torch.float64
    raise ValueError(f"Unsupported torch dtype. Got: {dtype}")

# Validate signature feature tensor for torch scoring
def validate_torch_signature_feature_tensor(
    features: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    squeezed = False
    x = features

    if x.ndim == 2:
        x = x.unsqueeze(0)
        squeezed = True
    elif x.ndim != 3:
        raise ValueError(
            "Signature features must have shape (P, D) or (B, P, D). "
            f"Got shape: {tuple(x.shape)}"
        )

    if x.shape[1] == 0:
        raise ValueError("Feature tensor must contain at least one prefix row.")
    if x.shape[2] == 0:
        raise ValueError("Feature tensor must contain at least one feature channel.")
    if not torch.isfinite(x).all():
        raise ValueError("Feature tensor contains NaN or infinite value(s).")

    return x, squeezed

# Store a torch linear stopping policy as a trainable module
class TorchLinearStoppingPolicy(nn.Module):
    # Initialize torch linear stopping policy parameters
    def __init__(
        self,
        input_dim: int,
        *,
        include_bias: bool = False,
        scalar_term_in_signatures: bool = True,
        init: str = "small_normal",
        dtype: torch.dtype = torch.float32,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError(f"input_dim must be positive. Got: {input_dim}")

        resolved_device = resolve_torch_device(device)
        resolved_dtype = resolve_torch_dtype(dtype)

        self.weights = nn.Parameter(
            torch.empty(input_dim, dtype = resolved_dtype, device = resolved_device)
        )
        if include_bias:
            self.bias = nn.Parameter(
                torch.zeros((), dtype = resolved_dtype, device = resolved_device)
            )
        else:
            self.register_parameter("bias", None)

        self.use_bias = bool(include_bias)
        self.scalar_term_in_signatures = bool(scalar_term_in_signatures)
        self.init = str(init)

    # Get number of input features expected by the policy
    @property
    def input_dim(self) -> int:
        return int(self.weights.shape[0])

    # Compute policy scores from signature features
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x, squeezed = validate_torch_signature_feature_tensor(features)

        if x.shape[-1] != self.input_dim:
            raise ValueError(
                "Policy input dimension does not match feature tensor width. "
                f"Policy input_dim={self.input_dim}, feature_dim={x.shape[-1]}."
            )

        scores = torch.tensordot(x, self.weights, dims = ([-1], [0]))
        if self.use_bias and self.bias is not None:
            scores = scores + self.bias

        if not torch.isfinite(scores).all():
            raise ValueError("Policy scores contain NaN or infinite value(s).")

        return scores[0] if squeezed else scores

    # Convert torch policy parameters into the serializable policy dataclass
    def to_linear_policy(self) -> LinearStoppingPolicy:
        return LinearStoppingPolicy(
            weights = self.weights.detach().cpu().numpy().copy(),
            bias = 0.0 if self.bias is None else float(self.bias.detach().cpu().item()),
            use_bias = self.use_bias,
            scalar_term_in_signatures = self.scalar_term_in_signatures,
            init = self.init,
        )

# Initialize torch linear stopping policy
def initialize_torch_linear_policy(
    input_dim: int,
    *,
    include_bias: bool = False,
    init: str = "small_normal",
    weight_scale: float = 1.0e-2,
    seed: int | None = None,
    dtype: str | np.dtype | torch.dtype | type[np.floating] = torch.float32,
    device: str | torch.device | None = None,
    scalar_term_in_signatures: bool = True,
) -> TorchLinearStoppingPolicy:
    if init not in SUPPORTED_TORCH_POLICY_INITS:
        raise ValueError(
            f"Unsupported policy init {init!r}. "
            f"Expected one of: {sorted(SUPPORTED_TORCH_POLICY_INITS)}"
        )

    resolved_dtype = resolve_torch_dtype(dtype)
    resolved_device = resolve_torch_device(device)
    resolved_scale = float(weight_scale)

    if not np.isfinite(resolved_scale):
        raise ValueError(f"weight_scale must be finite. Got: {weight_scale}")
    if resolved_scale < 0.0:
        raise ValueError(f"weight_scale must be non-negative. Got: {weight_scale}")

    module = TorchLinearStoppingPolicy(
        input_dim,
        include_bias = include_bias,
        scalar_term_in_signatures = scalar_term_in_signatures,
        init = init,
        dtype = resolved_dtype,
        device = resolved_device,
    )

    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))

    with torch.no_grad():
        if init == "small_normal":
            module.weights.normal_(mean = 0.0, std = resolved_scale)
        else:
            module.weights.zero_()

        if module.bias is not None:
            module.bias.zero_()

    return module

# Initialize torch policy from a feature tensor shape
def initialize_torch_policy_from_feature_tensor(
    features: np.ndarray | torch.Tensor | list[list[float]] | list[list[list[float]]],
    *,
    include_bias: bool = False,
    init: str = "small_normal",
    weight_scale: float = 1.0e-2,
    seed: int | None = None,
    dtype: str | np.dtype | torch.dtype | type[np.floating] = torch.float32,
    device: str | torch.device | None = None,
    scalar_term_in_signatures: bool = True,
) -> TorchLinearStoppingPolicy:
    input_dim = int(np.asarray(features).shape[-1])
    return initialize_torch_linear_policy(
        input_dim,
        include_bias = include_bias,
        init = init,
        weight_scale = weight_scale,
        seed = seed,
        dtype = dtype,
        device = device,
        scalar_term_in_signatures = scalar_term_in_signatures,
    )

# Build a torch policy module from the serializable policy dataclass
def build_torch_policy_from_linear_policy(
    policy: LinearStoppingPolicy,
    *,
    device: str | torch.device | None = None,
    dtype: str | np.dtype | torch.dtype | type[np.floating] | None = None,
) -> TorchLinearStoppingPolicy:
    resolved_dtype = resolve_torch_dtype(policy.weights.dtype if dtype is None else dtype)
    module = TorchLinearStoppingPolicy(
        policy.input_dim,
        include_bias = policy.use_bias,
        scalar_term_in_signatures = policy.scalar_term_in_signatures,
        init = policy.init,
        dtype = resolved_dtype,
        device = device,
    )

    with torch.no_grad():
        module.weights.copy_(torch.as_tensor(policy.weights, dtype = resolved_dtype, device = module.weights.device))
        if module.bias is not None:
            module.bias.fill_(float(policy.bias))

    return module
