from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

POWER_START = "19810101"


def cache_dir() -> Path:
    return Path(os.environ.get("MW_CACHE_DIR", str(DEFAULT_CACHE_DIR)))


@dataclass(frozen=True)
class City:
    key: str
    name: str
    country: str
    lat: float
    lon: float
    station: str
    unit: str = "C"
    currency: str = "$"
    group: str = "Custom"

    @property
    def base_temp(self) -> float:
        return 65.0 if self.unit == "F" else 18.0

    @property
    def unit_label(self) -> str:
        return "°F" if self.unit == "F" else "°C"

    @property
    def label(self) -> str:
        return f"{self.name}, {self.country}" if self.country else self.name


CITIES: tuple[City, ...] = (
    City("chicago", "Chicago", "USA", 41.9786, -87.9048, "O'Hare Intl (ORD)", "F", "$", "United States"),
    City("new_york", "New York", "USA", 40.7769, -73.8740, "LaGuardia (LGA)", "F", "$", "United States"),
    City("atlanta", "Atlanta", "USA", 33.6407, -84.4277, "Hartsfield-Jackson (ATL)", "F", "$", "United States"),
    City("dallas", "Dallas", "USA", 32.8998, -97.0403, "Dallas/Fort Worth (DFW)", "F", "$", "United States"),
    City("minneapolis", "Minneapolis", "USA", 44.8848, -93.2223, "Minneapolis-St Paul (MSP)", "F", "$", "United States"),
    City("cincinnati", "Cincinnati", "USA", 39.0489, -84.6678, "Cincinnati/N. Kentucky (CVG)", "F", "$", "United States"),
    City("las_vegas", "Las Vegas", "USA", 36.0840, -115.1537, "Harry Reid Intl (LAS)", "F", "$", "United States"),
    City("sacramento", "Sacramento", "USA", 38.5125, -121.4935, "Sacramento Executive (SAC)", "F", "$", "United States"),
    City("london", "London", "UK", 51.4700, -0.4543, "Heathrow (LHR)", "C", "£", "Europe"),
    City("amsterdam", "Amsterdam", "Netherlands", 52.3105, 4.7683, "Schiphol (AMS)", "C", "€", "Europe"),
    City("paris", "Paris", "France", 49.0097, 2.5479, "Charles de Gaulle (CDG)", "C", "€", "Europe"),
    City("berlin", "Berlin", "Germany", 52.3667, 13.5033, "Brandenburg (BER)", "C", "€", "Europe"),
    City("warsaw", "Warsaw", "Poland", 52.1657, 20.9671, "Chopin (WAW)", "C", "€", "Europe"),
    City("kyiv", "Kyiv", "Ukraine", 50.4501, 30.5234, "City centre", "C", "$", "Ukraine"),
    City("dnipro", "Dnipro", "Ukraine", 48.4647, 35.0462, "City centre", "C", "$", "Ukraine"),
    City("lviv", "Lviv", "Ukraine", 49.8397, 24.0297, "City centre", "C", "$", "Ukraine"),
    City("kharkiv", "Kharkiv", "Ukraine", 49.9935, 36.2304, "City centre", "C", "$", "Ukraine"),
    City("odesa", "Odesa", "Ukraine", 46.4825, 30.7233, "City centre", "C", "$", "Ukraine"),
)

CITY_BY_KEY: dict[str, City] = {city.key: city for city in CITIES}


@dataclass(frozen=True)
class ContractPeriod:
    key: str
    label: str
    index: str
    months: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.index not in ("HDD", "CDD"):
            raise ValueError(f"unknown index {self.index!r}")
        if not self.months or any((b - a) % 12 != 1 for a, b in zip(self.months, self.months[1:])):
            raise ValueError("contract months must be consecutive")

    @property
    def wraps_year(self) -> bool:
        return any(month < self.months[0] for month in self.months)

    @property
    def demand(self) -> str:
        return "Heating" if self.index == "HDD" else "Cooling"

    def season_label(self, season: int) -> str:
        if self.wraps_year:
            return f"{season}/{(season + 1) % 100:02d}"
        return str(season)


CONTRACTS: tuple[ContractPeriod, ...] = (
    ContractPeriod("winter_hdd", "Winter HDD (Nov–Mar)", "HDD", (11, 12, 1, 2, 3)),
    ContractPeriod("dec_hdd", "December HDD", "HDD", (12,)),
    ContractPeriod("jan_hdd", "January HDD", "HDD", (1,)),
    ContractPeriod("feb_hdd", "February HDD", "HDD", (2,)),
    ContractPeriod("summer_cdd", "Summer CDD (May–Sep)", "CDD", (5, 6, 7, 8, 9)),
    ContractPeriod("jul_cdd", "July CDD", "CDD", (7,)),
    ContractPeriod("aug_cdd", "August CDD", "CDD", (8,)),
)

CONTRACT_BY_KEY: dict[str, ContractPeriod] = {contract.key: contract for contract in CONTRACTS}
