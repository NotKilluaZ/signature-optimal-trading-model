from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen = True)
class BacktestCostConfig:
    fixed_per_action: float = 0.001
    proportional: float = 0.0005
    slippage_bps: float = 0.0
    model: str = "spread_proxy"

    # Validate backtest cost settings after initialization
    def __post_init__(self) -> None:
        if not np.isfinite(float(self.fixed_per_action)) or float(self.fixed_per_action) < 0.0:
            raise ValueError(
                f"fixed_per_action must be finite and non-negative. Got: {self.fixed_per_action}"
            )
        if not np.isfinite(float(self.proportional)) or float(self.proportional) < 0.0:
            raise ValueError(
                f"proportional must be finite and non-negative. Got: {self.proportional}"
            )
        if not np.isfinite(float(self.slippage_bps)) or float(self.slippage_bps) < 0.0:
            raise ValueError(
                f"slippage_bps must be finite and non-negative. Got: {self.slippage_bps}"
            )
        if str(self.model) not in {"spread_proxy", "leg_notional_proxy"}:
            raise ValueError(
                "model must be one of {'spread_proxy', 'leg_notional_proxy'}. "
                f"Got: {self.model!r}"
            )


# Read nested config values with a default fallback
def _get_config_value(config: dict[str, Any], path: list[str], default: Any = None) -> Any:
    current: Any = config
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


# Build the backtest cost config from the YAML config dictionary
def build_backtest_cost_config(config: dict[str, Any]) -> BacktestCostConfig:
    return BacktestCostConfig(
        fixed_per_action = float(_get_config_value(config, ["costs", "fixed_per_action"], 0.001)),
        proportional = float(_get_config_value(config, ["costs", "proportional"], 0.0005)),
        slippage_bps = float(_get_config_value(config, ["costs", "slippage_bps"], 0.0)),
        model = str(_get_config_value(config, ["costs", "model"], "spread_proxy")),
    )


# Compute one entry or exit cost under the configured cost model
def compute_action_cost(
    *,
    reference_spread: float,
    cost_config: BacktestCostConfig,
    notional_reference: float | None = None,
) -> float:
    spread_value = float(reference_spread)
    if not np.isfinite(spread_value):
        raise ValueError(f"reference_spread must be finite. Got: {reference_spread}")

    if cost_config.model == "spread_proxy":
        scale = abs(spread_value)
    else:
        if notional_reference is None:
            raise ValueError("notional_reference is required for model='leg_notional_proxy'.")
        scale = abs(float(notional_reference))
        if not np.isfinite(scale):
            raise ValueError(f"notional_reference must be finite. Got: {notional_reference}")

    slippage = scale * float(cost_config.slippage_bps) * 1.0e-4
    cost = float(cost_config.fixed_per_action) + float(cost_config.proportional) * scale + slippage
    if cost < 0.0 or not np.isfinite(cost):
        raise ValueError(f"Computed action cost must be finite and non-negative. Got: {cost}")
    return cost
