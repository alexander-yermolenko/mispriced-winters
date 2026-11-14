from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
OMEGA = 2.0 * np.pi / DAYS_PER_YEAR


def _days_since(dates, t0: pd.Timestamp) -> np.ndarray:
    delta = pd.DatetimeIndex(dates) - t0
    return np.asarray(delta / pd.Timedelta(days=1), dtype=float)


def design_matrix(t_days, harmonics: int = 3, seasonal_trend: bool = True) -> np.ndarray:
    t = np.asarray(t_days, dtype=float)
    years = t / DAYS_PER_YEAR
    columns = [np.ones_like(t), years]
    for k in range(1, harmonics + 1):
        columns.append(np.cos(k * OMEGA * t))
        columns.append(np.sin(k * OMEGA * t))
    if seasonal_trend:
        columns.append(years * np.cos(OMEGA * t))
        columns.append(years * np.sin(OMEGA * t))
    return np.column_stack(columns)


def is_stationary(phi) -> bool:
    phi = np.asarray(phi, dtype=float)
    p = phi.size
    if p == 0:
        return True
    companion = np.zeros((p, p))
    companion[0, :] = phi
    if p > 1:
        companion[1:, :-1] = np.eye(p - 1)
    return bool(np.max(np.abs(np.linalg.eigvals(companion))) < 1.0)


@dataclass
class TemperatureModel:
    t0: pd.Timestamp
    harmonics: int
    seasonal_trend: bool
    beta: np.ndarray
    phi: np.ndarray
    sigma: np.ndarray
    shocks: dict[int, np.ndarray]
    r2: float
    residual_mean_by_month: np.ndarray
    n_obs: int
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp

    @property
    def ar_order(self) -> int:
        return int(self.phi.size)

    @property
    def persistence(self) -> float:
        return float(np.sum(self.phi))

    @property
    def half_life_days(self) -> float:
        rho = self.persistence
        return float(np.log(0.5) / np.log(rho)) if 0.0 < rho < 1.0 else float("nan")

    def mean(self, dates) -> np.ndarray:
        t = _days_since(dates, self.t0)
        return design_matrix(t, self.harmonics, self.seasonal_trend) @ self.beta

    def warming_rate(self, dates) -> float:
        t = _days_since(dates, self.t0)
        rate = np.full_like(t, self.beta[1])
        if self.seasonal_trend:
            rate = rate + self.beta[-2] * np.cos(OMEGA * t) + self.beta[-1] * np.sin(OMEGA * t)
        return float(rate.mean() * 10.0)

    def simulate(
        self,
        dates,
        n_paths: int,
        rng: np.random.Generator,
        *,
        burn_in: int = 60,
        innovations: str = "bootstrap",
    ) -> np.ndarray:
        dates = pd.DatetimeIndex(dates)
        if len(dates) == 0:
            raise ValueError("no dates to simulate")
        if (dates[-1] - dates[0]).days + 1 != len(dates):
            raise ValueError("dates must be consecutive days")
        if innovations not in ("bootstrap", "gaussian"):
            raise ValueError(f"unknown innovations {innovations!r}")

        full = pd.date_range(dates[0] - pd.Timedelta(days=burn_in), dates[-1], freq="D")
        mu = self.mean(full)
        months = np.asarray(full.month)
        p = self.ar_order
        state = np.zeros((n_paths, p))
        anomalies = np.empty((n_paths, len(full)))
        for i, month in enumerate(months):
            if innovations == "bootstrap":
                pool = self.shocks[int(month)]
                z = pool[rng.integers(0, pool.size, size=n_paths)]
            else:
                z = rng.standard_normal(n_paths)
            x = state @ self.phi + self.sigma[month] * z
            state = np.roll(state, 1, axis=1)
            state[:, 0] = x
            anomalies[:, i] = x
        return mu[None, burn_in:] + anomalies[:, burn_in:]


def _fit_ar(anomaly: pd.Series, order: int) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    lagged = {f"lag{k}": anomaly.shift(k) for k in range(1, order + 1)}
    data = pd.concat({"y": anomaly, **lagged}, axis=1).dropna()
    z = data[[f"lag{k}" for k in range(1, order + 1)]].to_numpy()
    y = data["y"].to_numpy()
    phi, *_ = np.linalg.lstsq(z, y, rcond=None)
    return phi, y - z @ phi, pd.DatetimeIndex(data.index)


def fit_temperature_model(
    tavg_c: pd.Series,
    *,
    ar_order: int = 3,
    harmonics: int = 3,
    seasonal_trend: bool = True,
) -> TemperatureModel:
    series = tavg_c.dropna().sort_index()
    if len(series) < 3 * 365:
        raise ValueError("need at least three years of daily data to fit the model")
    if ar_order < 1:
        raise ValueError("ar_order must be at least 1")

    t0 = pd.Timestamp(series.index[0])
    x = design_matrix(_days_since(series.index, t0), harmonics, seasonal_trend)
    y = series.to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    residual = y - x @ beta
    r2 = float(1.0 - residual.var() / y.var())

    anomaly = pd.Series(residual, index=series.index).asfreq("D")
    phi, eps, eps_index = _fit_ar(anomaly, ar_order)
    if not is_stationary(phi):
        phi, eps, eps_index = _fit_ar(anomaly, 1)

    eps_months = np.asarray(eps_index.month)
    anomaly_valid = anomaly.dropna()
    anomaly_months = np.asarray(anomaly_valid.index.month)
    sigma = np.full(13, np.nan)
    residual_mean = np.full(13, np.nan)
    shocks: dict[int, np.ndarray] = {}
    for month in range(1, 13):
        e = eps[eps_months == month]
        sigma[month] = e.std(ddof=1)
        shocks[month] = (e - e.mean()) / sigma[month]
        residual_mean[month] = anomaly_valid.to_numpy()[anomaly_months == month].mean()

    return TemperatureModel(
        t0=t0,
        harmonics=harmonics,
        seasonal_trend=seasonal_trend,
        beta=beta,
        phi=np.asarray(phi, dtype=float),
        sigma=sigma,
        shocks=shocks,
        r2=r2,
        residual_mean_by_month=residual_mean,
        n_obs=int(series.size),
        fit_start=pd.Timestamp(series.index[0]),
        fit_end=pd.Timestamp(series.index[-1]),
    )
