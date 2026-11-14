from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .config import ContractPeriod

MAX_FILL_GAP_DAYS = 3


def _fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    missing = series.isna()
    if not missing.any():
        return series
    run_id = (missing != missing.shift()).cumsum()
    run_length = missing.groupby(run_id).transform("sum")
    fillable = missing & (run_length <= max_gap)
    interpolated = series.interpolate(limit_area="inside")
    return series.where(~fillable, interpolated)


def clean_daily(frame: pd.DataFrame, max_gap: int = MAX_FILL_GAP_DAYS) -> pd.DataFrame:
    frame = frame.sort_index()
    full = pd.date_range(frame.index.min(), frame.index.max(), freq="D", name="date")
    out = frame.reindex(full).apply(lambda column: _fill_short_gaps(column, max_gap))
    valid = out.dropna(how="any")
    if valid.empty:
        raise ValueError("no complete daily records")
    return out.loc[valid.index.min() : valid.index.max()]


def to_unit(temp_c, unit: str):
    if unit == "F":
        return temp_c * 9.0 / 5.0 + 32.0
    if unit == "C":
        return temp_c
    raise ValueError(f"unknown unit {unit!r}")


def daily_average_c(frame: pd.DataFrame) -> pd.Series:
    return ((frame["T2M_MAX"] + frame["T2M_MIN"]) / 2.0).rename("tavg")


def degree_days(temp, base: float, index: str):
    if index == "HDD":
        return np.maximum(base - temp, 0.0)
    if index == "CDD":
        return np.maximum(temp - base, 0.0)
    raise ValueError(f"unknown index {index!r}")


def season_of(dates: pd.DatetimeIndex, period: ContractPeriod) -> np.ndarray:
    years = np.asarray(dates.year)
    if period.wraps_year:
        return np.where(np.asarray(dates.month) >= period.months[0], years, years - 1)
    return years


def period_dates(season: int, period: ContractPeriod) -> pd.DatetimeIndex:
    start = pd.Timestamp(year=season, month=period.months[0], day=1)
    end_year = season + 1 if period.wraps_year else season
    end = pd.Timestamp(year=end_year, month=period.months[-1], day=1) + pd.offsets.MonthEnd(0)
    return pd.date_range(start, end, freq="D")


def period_frame(temp: pd.Series, period: ContractPeriod) -> pd.DataFrame:
    inside = temp[np.isin(np.asarray(temp.index.month), period.months)]
    return pd.DataFrame(
        {"temp": inside.to_numpy(dtype=float), "season": season_of(inside.index, period)},
        index=inside.index,
    )


def complete_seasons(frame: pd.DataFrame, period: ContractPeriod) -> list[int]:
    counts = frame["temp"].notna().groupby(frame["season"]).sum()
    return [int(season) for season, n in counts.items() if n == len(period_dates(int(season), period))]


def seasonal_index(
    frame: pd.DataFrame,
    period: ContractPeriod,
    base: float,
    shift: Mapping[int, float] | None = None,
) -> pd.Series:
    seasons = complete_seasons(frame, period)
    sub = frame[frame["season"].isin(seasons)]
    temp = sub["temp"].to_numpy(dtype=float)
    if shift:
        temp = temp + sub["season"].map(dict(shift)).fillna(0.0).to_numpy(dtype=float)
    daily = degree_days(temp, base, period.index)
    out = pd.Series(daily, index=sub.index).groupby(sub["season"].to_numpy()).sum()
    out.index = out.index.astype(int)
    out.index.name = "season"
    return out.rename("index")


def seasonal_mean_temp(frame: pd.DataFrame, period: ContractPeriod) -> pd.Series:
    seasons = complete_seasons(frame, period)
    sub = frame[frame["season"].isin(seasons)]
    out = sub.groupby("season")["temp"].mean()
    out.index = out.index.astype(int)
    return out.rename("mean_temp")
