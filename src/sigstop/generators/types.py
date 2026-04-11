from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any
import numpy as np


@dataclass(frozen = True)
class OUGeneratorParams:
    kappa: float
    theta: float
    sigma: float
    dt: float
    phi: float
    half_life_days: float
    estimation_method: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen = True)
class OUFitDiagnostics:
    log_likelihood: float
    neg_log_likelihood: float
    n_obs: int
    success: bool
    message: str
    var_floor: float
    fit_window: str = "formation"
    optimizer_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen = True)
class OUSampleRequest:
    x0: float
    horizon: int
    n_paths: int
    seed: int
    dt: float
    dtype: str
    device: str = "cpu"
    include_innovations: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OUSampleBatch:
    paths: np.ndarray
    metadata: dict[str, Any]
    innovations: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metadata": dict(self.metadata),
            "paths_shape": list(self.paths.shape),
            "paths_dtype": str(self.paths.dtype),
        }
        if self.innovations is not None:
            payload["innovations_shape"] = list(self.innovations.shape)
            payload["innovations_dtype"] = str(self.innovations.dtype)
        return payload


@dataclass(frozen = True)
class OUCurveErrorSummary:
    max_abs_error: float | None = None
    mean_abs_error: float | None = None
    max_rel_error: float | None = None
    mean_rel_error: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OUSampleValidationReport:
    status: str
    checks: dict[str, bool]
    notes: list[str] = field(default_factory = list)
    innovation_summary: dict[str, Any] | None = None
    mean_curve_error: OUCurveErrorSummary | None = None
    variance_curve_error: OUCurveErrorSummary | None = None
    acf_summary: dict[str, Any] | None = None
    empirical_half_life_summary: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "checks": dict(self.checks),
            "notes": list(self.notes),
        }
        if self.innovation_summary is not None:
            payload["innovation_summary"] = self.innovation_summary
        if self.mean_curve_error is not None:
            payload["mean_curve_error"] = self.mean_curve_error.to_dict()
        if self.variance_curve_error is not None:
            payload["variance_curve_error"] = self.variance_curve_error.to_dict()
        if self.acf_summary is not None:
            payload["acf_summary"] = self.acf_summary
        if self.empirical_half_life_summary is not None:
            payload["empirical_half_life_summary"] = self.empirical_half_life_summary
        return payload
