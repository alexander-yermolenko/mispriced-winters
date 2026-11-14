from __future__ import annotations

import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .config import POWER_START, cache_dir

POWER_DAILY_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
PARAMETERS = ("T2M", "T2M_MAX", "T2M_MIN")
USER_AGENT = "mispriced-winters/0.1 (NASA Space Apps Challenge 2026)"
RETRY_STATUS = (429, 500, 502, 503, 504)


class PowerAPIError(RuntimeError):
    """NASA POWER returned an error or an unexpected payload."""


def request_params(lat: float, lon: float, start: str = POWER_START, end: str | None = None) -> dict[str, str]:
    if end is None:
        end = (date.today() - timedelta(days=2)).strftime("%Y%m%d")
    return {
        "parameters": ",".join(PARAMETERS),
        "community": "SB",
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start": start,
        "end": end,
        "format": "JSON",
    }


def parse_power_json(payload: dict[str, Any]) -> pd.DataFrame:
    try:
        series = payload["properties"]["parameter"]
    except (KeyError, TypeError):
        detail = payload.get("messages") or payload.get("detail") or payload if isinstance(payload, dict) else payload
        raise PowerAPIError(f"Unexpected NASA POWER response: {str(detail)[:300]}") from None

    missing = [name for name in PARAMETERS if name not in series]
    if missing:
        raise PowerAPIError(f"NASA POWER response lacks parameters {missing}")

    header = payload.get("header") or {}
    fill_value = float(header.get("fill_value", header.get("fillValue", -999.0)))

    frame = pd.DataFrame({name: pd.Series(series[name], dtype=float) for name in PARAMETERS})
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame.index.name = "date"
    frame = frame.sort_index()
    return frame.where(frame > fill_value + 1.0)


def fetch_daily(
    lat: float,
    lon: float,
    *,
    start: str = POWER_START,
    end: str | None = None,
    session: Any = None,
    retries: int = 3,
    timeout: float = 180.0,
    backoff: float = 2.0,
) -> pd.DataFrame:
    params = request_params(lat, lon, start, end)
    http = session or requests.Session()
    headers = {"User-Agent": USER_AGENT}
    last_error: Exception | None = None
    shortened = False

    for attempt in range(retries):
        try:
            response = http.get(POWER_DAILY_URL, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
        else:
            if response.status_code == 200:
                return parse_power_json(response.json())
            message = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code == 422 and not shortened:
                params["end"] = (date.today() - timedelta(days=45)).strftime("%Y%m%d")
                shortened = True
                last_error = PowerAPIError(message)
                continue
            if response.status_code not in RETRY_STATUS:
                raise PowerAPIError(message)
            last_error = PowerAPIError(message)
        if attempt < retries - 1:
            time.sleep(backoff * 2**attempt)

    raise PowerAPIError(f"NASA POWER request failed after {retries} attempts: {last_error}")


def cache_path(key: str) -> Path:
    return cache_dir() / f"{key}.csv"


def is_cached(key: str) -> bool:
    return cache_path(key).exists()


def load_cached(key: str) -> pd.DataFrame | None:
    path = cache_path(key)
    if not path.exists():
        return None
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    return frame.astype(float)


def save_cached(key: str, frame: pd.DataFrame) -> Path:
    path = cache_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, float_format="%.2f", index_label="date")
    return path


def get_daily(lat: float, lon: float, key: str, *, refresh: bool = False, **fetch_kwargs: Any) -> pd.DataFrame:
    if not refresh:
        cached = load_cached(key)
        if cached is not None:
            return cached
    frame = fetch_daily(lat, lon, **fetch_kwargs)
    save_cached(key, frame)
    return frame
