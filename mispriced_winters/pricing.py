from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class OptionSpec:
    kind: str
    strike: float
    tick: float
    cap: float | None = None
    rate: float = 0.0
    tau: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("call", "put"):
            raise ValueError(f"unknown option kind {self.kind!r}")
        if self.tick <= 0:
            raise ValueError("tick must be positive")

    @property
    def discount_factor(self) -> float:
        return float(np.exp(-self.rate * self.tau))


def payoff(index_values, spec: OptionSpec) -> np.ndarray:
    values = np.asarray(index_values, dtype=float)
    if spec.kind == "call":
        intrinsic = np.maximum(values - spec.strike, 0.0)
    else:
        intrinsic = np.maximum(spec.strike - values, 0.0)
    cash = spec.tick * intrinsic
    if spec.cap is not None and spec.cap > 0:
        cash = np.minimum(cash, spec.cap)
    return cash


@dataclass(frozen=True)
class PriceResult:
    method: str
    price: float
    stderr: float
    expected_index: float
    index_std: float
    n: int


def expected_payoff_price(method: str, index_values, spec: OptionSpec) -> PriceResult:
    values = np.asarray(index_values, dtype=float)
    if values.size == 0:
        raise ValueError("no index values to price on")
    cash = payoff(values, spec)
    discount = spec.discount_factor
    n = values.size
    stderr = discount * cash.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
    index_std = values.std(ddof=1) if n > 1 else float("nan")
    return PriceResult(method, float(discount * cash.mean()), float(stderr), float(values.mean()), float(index_std), int(n))


def bachelier_price(mean: float, std: float, spec: OptionSpec) -> float:
    if spec.cap:
        raise ValueError("closed form is only available for uncapped options")
    d = (mean - spec.strike) / std
    if spec.kind == "call":
        value = (mean - spec.strike) * stats.norm.cdf(d) + std * stats.norm.pdf(d)
    else:
        value = (spec.strike - mean) * stats.norm.cdf(-d) + std * stats.norm.pdf(d)
    return float(spec.discount_factor * spec.tick * value)
