from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class TrendResult:
    n: int
    s: float
    var_s: float
    z: float
    p_value: float
    tau: float
    slope: float
    intercept: float
    hamed_rao: bool

    def significant(self, alpha: float = 0.05) -> bool:
        return bool(self.p_value < alpha)

    def direction(self, alpha: float = 0.05) -> str:
        if not self.significant(alpha):
            return "no significant trend"
        return "increasing" if self.z > 0 else "decreasing"


def _as_xy(y, x) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.arange(y.size, dtype=float) if x is None else np.asarray(x, dtype=float)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same length")
    keep = ~(np.isnan(y) | np.isnan(x))
    y, x = y[keep], x[keep]
    order = np.argsort(x, kind="stable")
    return y[order], x[order]


def sens_slope(y, x=None) -> tuple[float, float]:
    y, x = _as_xy(y, x)
    if y.size < 2:
        return float("nan"), float("nan")
    i, j = np.triu_indices(y.size, k=1)
    dx = x[j] - x[i]
    valid = dx != 0
    slope = float(np.median((y[j] - y[i])[valid] / dx[valid]))
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def _mk_score(y: np.ndarray) -> float:
    i, j = np.triu_indices(y.size, k=1)
    return float(np.sign(y[j] - y[i]).sum())


def _mk_variance(y: np.ndarray) -> float:
    n = y.size
    _, counts = np.unique(y, return_counts=True)
    ties = counts[counts > 1].astype(float)
    return float((n * (n - 1) * (2 * n + 5) - np.sum(ties * (ties - 1) * (2 * ties + 5))) / 18.0)


def _autocorrelation(values: np.ndarray, nlags: int) -> np.ndarray:
    centred = values - values.mean()
    n = centred.size
    acov = np.correlate(centred, centred, mode="full")[n - 1 :] / n
    return acov[: nlags + 1] / acov[0]


def mann_kendall(y, x=None, *, hamed_rao: bool = True, alpha: float = 0.05) -> TrendResult:
    y, x = _as_xy(y, x)
    n = y.size
    if n < 4:
        raise ValueError("Mann–Kendall needs at least 4 observations")

    s = _mk_score(y)
    var_s = _mk_variance(y)
    slope, intercept = sens_slope(y, x)

    if hamed_rao:
        ranks = stats.rankdata(y - slope * x)
        rho = _autocorrelation(ranks, n - 1)[1:]
        lags = np.arange(1, n)
        bound = stats.norm.ppf(1.0 - alpha / 2.0) / np.sqrt(n)
        keep = np.abs(rho) > bound
        weights = (n - lags) * (n - lags - 1) * (n - lags - 2)
        factor = 1.0 + 2.0 / (n * (n - 1) * (n - 2)) * float(np.sum(weights[keep] * rho[keep]))
        if factor > 0:
            var_s *= factor

    if s > 0:
        z = (s - 1.0) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1.0) / np.sqrt(var_s)
    else:
        z = 0.0
    p_value = float(2.0 * stats.norm.sf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    return TrendResult(n, s, var_s, float(z), p_value, float(tau), slope, intercept, hamed_rao)


def benjamini_hochberg(p_values) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if m == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    scaled = pv[order] * m / np.arange(1, m + 1)
    q_sorted = np.minimum.accumulate(scaled[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    out[ok] = q
    return out
