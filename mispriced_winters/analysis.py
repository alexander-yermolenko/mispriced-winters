from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import City, ContractPeriod
from .degree_days import (
    clean_daily,
    daily_average_c,
    degree_days,
    period_dates,
    period_frame,
    seasonal_index,
    seasonal_mean_temp,
    to_unit,
)
from .pricing import OptionSpec, PriceResult, expected_payoff_price, payoff
from .temperature_model import TemperatureModel, fit_temperature_model
from .trends import TrendResult, mann_kendall, sens_slope

MIN_SEASONS = 15


@dataclass(frozen=True)
class PricingInputs:
    kind: str = "call"
    strike_offset: float = 0.0
    tick: float = 20.0
    cap: float = 0.0
    rate: float = 0.0
    lookback: int = 20

    def spec(self, strike: float, tau: float = 0.0) -> OptionSpec:
        return OptionSpec(self.kind, strike, self.tick, self.cap or None, self.rate, tau)


@dataclass(frozen=True)
class ModelInputs:
    n_paths: int = 5000
    ar_order: int = 3
    harmonics: int = 3
    seasonal_trend: bool = True
    innovations: str = "bootstrap"
    match_volatility: bool = True
    seed: int = 42


@dataclass
class CityAnalysis:
    city: City
    period: ContractPeriod
    target_season: int
    index_history: pd.Series
    mean_temp_history: pd.Series
    detrended_history: pd.Series
    index_trend: TrendResult
    temp_trend: TrendResult
    lookback_seasons: np.ndarray
    simulated_index: np.ndarray
    raw_simulated_std: float
    strike: float
    spec: OptionSpec
    prices: dict[str, PriceResult]
    model: TemperatureModel

    @property
    def naive_index(self) -> np.ndarray:
        return self.index_history.loc[self.lookback_seasons].to_numpy()

    @property
    def detrended_index(self) -> np.ndarray:
        return self.detrended_history.loc[self.lookback_seasons].to_numpy()

    @property
    def target_label(self) -> str:
        return self.period.season_label(self.target_season)

    @property
    def trend_per_decade(self) -> float:
        return self.index_trend.slope * 10.0

    @property
    def trend_pct_per_decade(self) -> float:
        return self.trend_per_decade / float(self.index_history.mean()) * 100.0

    @property
    def warming_c_per_decade(self) -> float:
        per_decade = self.temp_trend.slope * 10.0
        return per_decade / 1.8 if self.city.unit == "F" else per_decade

    @property
    def mispricing(self) -> float:
        return self.prices["naive"].price - self.prices["detrended"].price

    @property
    def mispricing_pct(self) -> float:
        return relative_gap(self.prices["naive"].price, self.prices["detrended"].price)

    @property
    def model_gap_pct(self) -> float:
        return relative_gap(self.prices["naive"].price, self.prices["model"].price)

    @property
    def index_bias(self) -> float:
        return self.prices["naive"].expected_index - self.prices["detrended"].expected_index

    @property
    def futures_mispricing(self) -> float:
        return self.spec.tick * self.index_bias

    @property
    def historical_detrended_std(self) -> float:
        return float(self.detrended_history.std(ddof=1))


def relative_gap(value: float, reference: float) -> float:
    return (value - reference) / reference * 100.0 if reference > 1e-9 else float("nan")


def prepare(daily: pd.DataFrame, city: City, period: ContractPeriod) -> tuple[pd.Series, pd.DataFrame]:
    tavg_c = daily_average_c(clean_daily(daily))
    frame = period_frame(to_unit(tavg_c, city.unit), period)
    return tavg_c, frame


def shifted_index(
    frame: pd.DataFrame,
    period: ContractPeriod,
    base: float,
    seasons,
    slope: float,
    target: int,
) -> pd.Series:
    seasons = [int(season) for season in seasons]
    shift = {season: slope * (target - season) for season in seasons}
    sub = frame[frame["season"].isin(seasons)]
    return seasonal_index(sub, period, base, shift=shift)


def analyze_city(
    daily: pd.DataFrame,
    city: City,
    period: ContractPeriod,
    pricing: PricingInputs = PricingInputs(),
    model_inputs: ModelInputs = ModelInputs(),
) -> CityAnalysis:
    tavg_c, frame = prepare(daily, city, period)
    base = city.base_temp
    history = seasonal_index(frame, period, base)
    if len(history) < MIN_SEASONS:
        raise ValueError(f"only {len(history)} complete seasons in the data; need at least {MIN_SEASONS}")
    mean_temp = seasonal_mean_temp(frame, period)

    seasons = history.index.to_numpy()
    target = int(seasons.max()) + 1
    lookback = int(min(max(pricing.lookback, 2), len(history)))
    lookback_seasons = seasons[-lookback:]

    index_trend = mann_kendall(history.to_numpy(), seasons)
    temp_trend = mann_kendall(mean_temp.to_numpy(), mean_temp.index.to_numpy())

    detrended_history = shifted_index(frame, period, base, seasons, temp_trend.slope, target)

    target_dates = period_dates(target, period)
    cutoff = target_dates[0] - pd.Timedelta(days=1)
    model = fit_temperature_model(
        tavg_c.loc[:cutoff],
        ar_order=model_inputs.ar_order,
        harmonics=model_inputs.harmonics,
        seasonal_trend=model_inputs.seasonal_trend,
    )
    rng = np.random.default_rng(model_inputs.seed)
    paths_c = model.simulate(target_dates, model_inputs.n_paths, rng, innovations=model_inputs.innovations)
    simulated = degree_days(to_unit(paths_c, city.unit), base, period.index).sum(axis=1)
    raw_std = float(simulated.std(ddof=1))
    if model_inputs.match_volatility and raw_std > 0:
        center = float(simulated.mean())
        scale = float(detrended_history.std(ddof=1)) / raw_std
        simulated = np.maximum(center + (simulated - center) * scale, 0.0)

    naive = history.loc[lookback_seasons].to_numpy()
    detrended = detrended_history.loc[lookback_seasons].to_numpy()
    strike = float(naive.mean() * (1.0 + pricing.strike_offset))
    spec = pricing.spec(strike, tau=len(target_dates) / 365.25)
    prices = {
        "naive": expected_payoff_price(f"Naive burn ({lookback} seasons)", naive, spec),
        "detrended": expected_payoff_price(f"Trend-adjusted burn ({lookback} seasons)", detrended, spec),
        "model": expected_payoff_price(f"Seasonal AR({model.ar_order}) Monte Carlo", simulated, spec),
    }

    return CityAnalysis(
        city=city,
        period=period,
        target_season=target,
        index_history=history,
        mean_temp_history=mean_temp,
        detrended_history=detrended_history,
        index_trend=index_trend,
        temp_trend=temp_trend,
        lookback_seasons=lookback_seasons,
        simulated_index=simulated,
        raw_simulated_std=raw_std,
        strike=strike,
        spec=spec,
        prices=prices,
        model=model,
    )


def summary_row(analysis: CityAnalysis) -> dict[str, Any]:
    city = analysis.city
    return {
        "key": city.key,
        "City": city.label,
        "Group": city.group,
        "Station": city.station,
        "lat": city.lat,
        "lon": city.lon,
        "Unit": city.unit_label,
        "Currency": city.currency,
        "Seasons": len(analysis.index_history),
        "Target season": analysis.target_label,
        "Mean index": float(analysis.index_history.mean()),
        "Trend / decade": analysis.trend_per_decade,
        "Trend % / decade": analysis.trend_pct_per_decade,
        "Warming °C / decade": analysis.warming_c_per_decade,
        "p_value": analysis.index_trend.p_value,
        "Strike": analysis.strike,
        "Naive": analysis.prices["naive"].price,
        "Trend-adjusted": analysis.prices["detrended"].price,
        "Model": analysis.prices["model"].price,
        "Mispricing %": analysis.mispricing_pct,
        "Naive vs model %": analysis.model_gap_pct,
        "Index bias": analysis.index_bias,
    }


def lookback_sensitivity(analysis: CityAnalysis, lookbacks=None) -> pd.DataFrame:
    n = len(analysis.index_history)
    if lookbacks is None:
        lookbacks = sorted(set(range(5, n + 1, 5)) | {n})
    rows = []
    for length in lookbacks:
        if length < 2 or length > n:
            continue
        naive = analysis.index_history.iloc[-length:].to_numpy()
        detrended = analysis.detrended_history.iloc[-length:].to_numpy()
        rows.append(
            {
                "lookback": int(length),
                "naive_price": expected_payoff_price("naive", naive, analysis.spec).price,
                "detrended_price": expected_payoff_price("detrended", detrended, analysis.spec).price,
                "naive_expected_index": float(naive.mean()),
                "detrended_expected_index": float(detrended.mean()),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class BacktestResult:
    table: pd.DataFrame
    summary: dict[str, float]


def _t_stat(values) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return float("nan")
    sd = values.std(ddof=1)
    return float(values.mean() / (sd / np.sqrt(values.size))) if sd > 0 else float("nan")


def walk_forward_backtest(
    daily: pd.DataFrame,
    city: City,
    period: ContractPeriod,
    pricing: PricingInputs = PricingInputs(),
    min_history: int = MIN_SEASONS,
) -> BacktestResult:
    _, frame = prepare(daily, city, period)
    base = city.base_temp
    history = seasonal_index(frame, period, base)
    mean_temp = seasonal_mean_temp(frame, period)
    seasons = history.index.to_numpy()
    lookback = max(int(pricing.lookback), 2)

    rows = []
    for pos in range(max(lookback, min_history), len(seasons)):
        season = int(seasons[pos])
        window = seasons[pos - lookback : pos]
        naive = history.loc[window].to_numpy()
        past_temp = mean_temp[mean_temp.index < season]
        slope, _ = sens_slope(past_temp.to_numpy(), past_temp.index.to_numpy())
        detrended = shifted_index(frame, period, base, window, slope, season).to_numpy()
        spec = pricing.spec(float(naive.mean() * (1.0 + pricing.strike_offset)))
        realized = float(history.loc[season])
        rows.append(
            {
                "season": season,
                "label": period.season_label(season),
                "strike": spec.strike,
                "realized_index": realized,
                "naive_forecast": float(naive.mean()),
                "detrended_forecast": float(detrended.mean()),
                "naive_price": float(payoff(naive, spec).mean()),
                "detrended_price": float(payoff(detrended, spec).mean()),
                "realized_payoff": float(payoff([realized], spec)[0]),
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        return BacktestResult(table, {"n": 0})
    table = table.set_index("season")
    table["naive_pnl"] = table["realized_payoff"] - table["naive_price"]
    table["detrended_pnl"] = table["realized_payoff"] - table["detrended_price"]

    naive_error = (table["naive_forecast"] - table["realized_index"]).to_numpy()
    detrended_error = (table["detrended_forecast"] - table["realized_index"]).to_numpy()
    summary = {
        "n": int(len(table)),
        "naive_index_bias": float(naive_error.mean()),
        "naive_index_bias_t": _t_stat(naive_error),
        "detrended_index_bias": float(detrended_error.mean()),
        "detrended_index_bias_t": _t_stat(detrended_error),
        "naive_index_rmse": float(np.sqrt(np.mean(naive_error**2))),
        "detrended_index_rmse": float(np.sqrt(np.mean(detrended_error**2))),
        "naive_premium_total": float(table["naive_price"].sum()),
        "detrended_premium_total": float(table["detrended_price"].sum()),
        "naive_buyer_pnl": float(table["naive_pnl"].sum()),
        "detrended_buyer_pnl": float(table["detrended_pnl"].sum()),
    }
    return BacktestResult(table, summary)
