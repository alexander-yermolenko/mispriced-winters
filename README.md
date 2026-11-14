# ❄️ Mispriced Winters

**Trend-adjusted pricing of weather derivatives with NASA POWER data.**
NASA Space Apps Challenge 2026 · Challenge: *Be an Earth System Trend Detective* · Team Space Alpha, Dnipro, Ukraine

Weather derivatives pay out on **heating and cooling degree days** (HDD/CDD). Many pricers still use
*burn analysis*: replay the last 10–30 seasons and average the payouts. If winters are getting warmer, the
past contains too many cold seasons, so the naive price of an HDD call is too high and the price of a CDD call
too low. Mispriced Winters uses four decades of NASA POWER daily temperatures to:

1. **detect** statistically significant trends in the settlement index of 18 cities (plus any point on Earth),
2. **price** the same option three ways — naive burn, trend-adjusted burn, and a seasonal stochastic
   temperature model with Monte Carlo,
3. **test** the mispricing with a walk-forward backtest that never looks ahead.

## Quick start

Requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/prefetch.py           # one request per city to NASA POWER; cached in data/cache/
streamlit run app.py
```

The app can also download data itself (buttons in the **Overview** and **City case file** tabs), and the
sidebar accepts a custom latitude/longitude.

**Tip for the demo:** run `prefetch.py` on good Wi-Fi beforehand and commit `data/cache/` to the repo, so the
app (and a Streamlit Community Cloud deployment) starts instantly without calling the API.

## How it answers the challenge

| Challenge question | Where in the app | Method |
|---|---|---|
| What is changing? | City case file → “What is changing?” | Seasonal HDD/CDD settlement index (Nov–Mar, May–Sep, single months) |
| Where is it changing? | Overview map + table | 8 US airport settlement stations, 5 European airports, 5 Ukrainian cities, custom points |
| How much? | “How much?” card, trend line | Theil–Sen slope, degree days and % per decade; season warming in °C/decade |
| Is it significant? | “Is it significant?” card, q-values | Mann–Kendall with Hamed–Rao autocorrelation correction; Benjamini–Hochberg FDR across cities |

## Method in one screen

* **Data** — NASA POWER daily `T2M_MAX` / `T2M_MIN` (MERRA-2 based), 1981 to today, grid cell around each
  station. Short gaps (≤ 3 days) are interpolated; incomplete seasons are dropped.
* **Index** — exchange convention: `Tavg = (Tmax + Tmin) / 2`, `HDD = max(0, base − Tavg)`,
  `CDD = max(0, Tavg − base)`, summed over the contract period. Base 65 °F for US-style contracts, 18 °C elsewhere.
* **Naive burn** — mean payout over the last *L* seasons, as they happened.
* **Trend-adjusted burn** — the same seasons with every day shifted by `β̂·(target − season)`, where β̂ is the
  Sen slope of the season's mean temperature; degree days are recomputed so truncation at the base stays exact.
* **Monte Carlo model** — `T(t) = a + b·t + Σ harmonics + t·(e·cos ωt + f·sin ωt) + X(t)`, with `X` an AR(p)
  process with a volatility per calendar month (Alaton, Djehiche & Stillberger, 2002) and bootstrapped shocks.
  The simulated index can be rescaled to the detrended historical variance (daily models understate it).
* **Walk-forward backtest** — every past season is priced using only earlier seasons, then compared with the
  realised index and payout.

Prices are actuarial (expected payoff, optionally discounted), with no risk loading.

## Project structure

```
app.py                         Streamlit front end (Overview · City case file · Backtest · Method)
mispriced_winters/
  config.py                    cities, contract periods, cache location (MW_CACHE_DIR)
  power_client.py              NASA POWER daily point API client + CSV cache
  degree_days.py               cleaning, HDD/CDD, seasons, season shifting
  trends.py                    Mann–Kendall (Hamed–Rao), Sen's slope, Benjamini–Hochberg
  temperature_model.py         seasonal mean + AR(p) with monthly volatility, simulation
  pricing.py                   payoffs, expected-payoff pricing, Bachelier check
  analysis.py                  per-city analysis, lookback sensitivity, walk-forward backtest
scripts/prefetch.py            download all cities ahead of the demo
data/cache/                    NASA POWER downloads (created on the first run)
requirements.txt               Python dependencies
.streamlit/config.toml         Streamlit settings and theme colour
```

## Caveats

* **Basis risk** — a reanalysis grid cell (~50 km) is not the settlement station's thermometer.
* **Day definition** — POWER's daily max/min may not follow the station's local climatological day.
* **Homogeneity** — reanalysis can contain artificial jumps; confirm findings against station data (e.g. NOAA GHCN-Daily).
* **Model scope** — linear trend, no seasonal forecasts (ENSO/NAO), no risk premium. Ukrainian contracts are hypothetical.

## Data and references

**NASA data.** Daily `T2M_MAX` and `T2M_MIN` from the
[NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), which is based on NASA's
[MERRA-2 reanalysis](https://gmao.gsfc.nasa.gov/reanalysis/merra-2/). These data were obtained from the
NASA Langley Research Center (LaRC) POWER Project, funded through the NASA Earth Science/Applied Science
Program — https://power.larc.nasa.gov

**Methods**

* Alaton, Djehiche & Stillberger (2002), *On modelling and pricing weather derivatives*, Applied Mathematical
  Finance 9(1). [doi:10.1080/13504860210132897](https://doi.org/10.1080/13504860210132897)
* Benth & Šaltytė Benth (2007), *The volatility of temperature and pricing of weather derivatives*, Quantitative Finance 7(5).
* Jewson & Brix (2005), *Weather Derivative Valuation*, Cambridge University Press.
* Mann (1945), *Nonparametric tests against trend*, Econometrica 13(3).
  [doi:10.2307/1907187](https://doi.org/10.2307/1907187)
* Kendall (1975), *Rank Correlation Methods*, Griffin, London.
* Sen (1968), *Estimates of the regression coefficient based on Kendall's tau*, Journal of the American
  Statistical Association 63. [doi:10.2307/2285891](https://doi.org/10.2307/2285891)
* Hamed & Rao (1998), *A modified Mann–Kendall trend test for autocorrelated data*, Journal of Hydrology 204.
  [doi:10.1016/S0022-1694(97)00125-X](https://doi.org/10.1016/S0022-1694(97)00125-X)
* Benjamini & Hochberg (1995), *Controlling the false discovery rate: a practical and powerful approach to
  multiple testing*, Journal of the Royal Statistical Society B 57.
  [doi:10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)

**Market and climate facts used in the presentation**

* CME Group — [Weather options overview](https://www.cmegroup.com/education/articles-and-reports/weather-options-overview.html)
  and [weather futures and options fact card](https://www.cmegroup.com/trading/weather/files/weather-fact-card.pdf).
* Copernicus Climate Change Service & WMO —
  [European State of the Climate 2025](https://climate.copernicus.eu/ten-charts-discover-european-state-climate-2025-report),
  as reported by [RTÉ](https://www.rte.ie/news/2026/0429/1570737-europe-climate-report/).

## Use of Super Intelligence (SI)

We used an SI assistant, Claude by Anthropic, throughout this project: to brainstorm the idea,
to generate most of the Python code, and to draft this README and the slides.
The temperature data is real NASA POWER data and is not SI-generated.
