from __future__ import annotations

from dataclasses import astuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from mispriced_winters import power_client
from mispriced_winters.analysis import (
    BacktestResult,
    CityAnalysis,
    ModelInputs,
    PricingInputs,
    analyze_city,
    lookback_sensitivity,
    summary_row,
    walk_forward_backtest,
)
from mispriced_winters.config import CITIES, CITY_BY_KEY, CONTRACT_BY_KEY, CONTRACTS, City, ContractPeriod
from mispriced_winters.trends import benjamini_hochberg

st.set_page_config(page_title="Mispriced Winters", page_icon="❄️", layout="wide")

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'
ACKNOWLEDGEMENT = (
    "Data: NASA Langley Research Center (LaRC) POWER Project, funded through the NASA Earth "
    "Science/Applied Science Program — daily 2 m temperature, 1981 to today."
)

LIGHT = {
    "naive": "#eb6834", "adjusted": "#2a78d6", "model": "#1baf7a",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "history": "#c3c2b7", "lookback": "#52514e",
    "cold": "#2a78d6", "neutral": "#f0efec", "warm": "#e34948", "land": "#f0efec", "surface": "#fcfcfb",
}
DARK = {
    "naive": "#d95926", "adjusted": "#3987e5", "model": "#199e70",
    "ink": "#f0efec", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "history": "#52514e", "lookback": "#c3c2b7",
    "cold": "#3987e5", "neutral": "#383835", "warm": "#e66767", "land": "#2c2c2a", "surface": "#1a1a19",
}


def palette() -> dict[str, str]:
    try:
        theme_type = st.context.theme.type
    except Exception:
        theme_type = None
    return DARK if theme_type == "dark" else LIGHT


def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def money(value: float, currency: str) -> str:
    if not np.isfinite(value):
        return "n/a"
    sign = "−" if value < 0 else ""
    return f"{sign}{currency}{abs(value):,.0f}"


def pct(value: float) -> str:
    return "n/a" if not np.isfinite(value) else f"{value:+.0f}%"


def style(fig: go.Figure, *, height: int = 380, x_title: str | None = None, y_title: str | None = None,
          x_grid: bool = False) -> go.Figure:
    p = palette()
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=36, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=p["ink2"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(color=p["ink2"])),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    )
    axis = dict(
        automargin=True, gridcolor=p["grid"], gridwidth=1, zeroline=False, showline=True, linecolor=p["axis"], linewidth=1,
        tickfont=dict(color=p["muted"]), title_font=dict(color=p["ink2"]),
    )
    fig.update_xaxes(showgrid=x_grid, title_text=x_title, **axis)
    fig.update_yaxes(showgrid=True, title_text=y_title, **axis)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config={"displaylogo": False})


def data_version(key: str) -> float:
    path = power_client.cache_path(key)
    return path.stat().st_mtime if path.exists() else 0.0


def custom_city(name: str, lat: float, lon: float, unit: str) -> City:
    key = f"custom_{lat:.2f}_{lon:.2f}"
    return City(key, name or "Custom location", "", float(lat), float(lon), "Custom point", unit, "$", "Custom")


@st.cache_data(show_spinner=False, max_entries=512)
def run_analysis(city_t: tuple, period_key: str, pricing_t: tuple, model_t: tuple, version: float) -> CityAnalysis:
    city = City(*city_t)
    daily = power_client.load_cached(city.key)
    if daily is None:
        raise FileNotFoundError(f"no cached NASA POWER data for {city.label}")
    return analyze_city(daily, city, CONTRACT_BY_KEY[period_key], PricingInputs(*pricing_t), ModelInputs(*model_t))


@st.cache_data(show_spinner=False, max_entries=128)
def run_backtest(city_t: tuple, period_key: str, pricing_t: tuple, version: float) -> BacktestResult:
    city = City(*city_t)
    daily = power_client.load_cached(city.key)
    if daily is None:
        raise FileNotFoundError(f"no cached NASA POWER data for {city.label}")
    return walk_forward_backtest(daily, city, CONTRACT_BY_KEY[period_key], PricingInputs(*pricing_t))


def download(cities: list[City]) -> list[str]:
    errors: list[str] = []
    progress = st.progress(0.0, text="Contacting NASA POWER…")
    for i, city in enumerate(cities, start=1):
        try:
            power_client.get_daily(city.lat, city.lon, city.key, refresh=True)
        except Exception as exc:
            errors.append(f"{city.label}: {exc}")
        progress.progress(i / len(cities), text=f"{i}/{len(cities)} · {city.label}")
    st.cache_data.clear()
    return errors


def sidebar() -> tuple[City, ContractPeriod, PricingInputs, ModelInputs]:
    with st.sidebar:
        st.header("Location")
        options = [city.key for city in CITIES] + ["custom"]
        default_key = next((city.key for city in CITIES if power_client.is_cached(city.key)), CITIES[0].key)
        key = st.selectbox(
            "City",
            options,
            index=options.index(default_key),
            key="city",
            format_func=lambda k: "Custom location…" if k == "custom" else f"{CITY_BY_KEY[k].label} · {CITY_BY_KEY[k].station}",
        )
        if key == "custom":
            name = st.text_input("Name", "My location", key="custom_name")
            lat = st.number_input("Latitude", -90.0, 90.0, 48.4647, 0.01, format="%.4f", key="custom_lat")
            lon = st.number_input("Longitude", -180.0, 180.0, 35.0462, 0.01, format="%.4f", key="custom_lon")
            unit = st.radio(
                "Contract convention", ["C", "F"], horizontal=True, key="custom_unit",
                format_func=lambda u: "°C, 18° base" if u == "C" else "°F, 65° base",
            )
            city = custom_city(name, lat, lon, unit)
        else:
            city = CITY_BY_KEY[key]

        st.header("Contract")
        period_key = st.selectbox(
            "Index and period", [c.key for c in CONTRACTS], key="contract",
            format_func=lambda k: CONTRACT_BY_KEY[k].label,
        )
        kind = st.radio("Option", ["call", "put"], horizontal=True, key="kind", format_func=str.title)
        lookback = st.slider(
            "Burn-analysis lookback (seasons)", 10, 40, 20, key="lookback",
            help="How many past seasons a naive pricer averages over.",
        )
        offset = st.slider(
            "Strike vs naive expected index (%)", -15, 15, 0, key="offset",
            help="0 = at the money for a pricer who ignores the trend.",
        )
        tick = st.number_input("Tick (currency per degree day)", 1.0, 10_000.0, 20.0, 1.0, key="tick")
        cap = st.number_input("Payout cap (0 = none)", 0.0, 100_000_000.0, 0.0, 1_000.0, key="cap")
        rate = st.number_input("Discount rate (% per year)", 0.0, 20.0, 0.0, 0.25, key="rate")

        with st.expander("Model settings"):
            n_paths = st.select_slider("Monte Carlo paths", [1_000, 2_000, 5_000, 10_000, 20_000], 5_000, key="paths")
            ar_order = st.select_slider("AR order of daily anomalies", [1, 2, 3], 3, key="ar")
            innovations = st.radio(
                "Daily shocks", ["bootstrap", "gaussian"], key="shocks",
                format_func=lambda s: "Bootstrap historical shocks" if s == "bootstrap" else "Gaussian",
            )
            match = st.toggle("Match simulated index volatility to history", True, key="match")
            seasonal_trend = st.toggle("Let the trend differ by season", True, key="season_trend")
        st.caption(ACKNOWLEDGEMENT)

    pricing = PricingInputs(kind, offset / 100.0, float(tick), float(cap), rate / 100.0, int(lookback))
    model = ModelInputs(int(n_paths), int(ar_order), 3, bool(seasonal_trend), innovations, bool(match), 42)
    return city, CONTRACT_BY_KEY[period_key], pricing, model


def overview_figures(df: pd.DataFrame, period: ContractPeriod, pricing: PricingInputs) -> tuple[go.Figure, go.Figure]:
    p = palette()
    warming = df["Warming °C / decade"].to_numpy(dtype=float)
    limit = max(0.1, float(np.nanmax(np.abs(warming))))
    custom = np.stack([df["Trend % / decade"], df["q_value"], df["Mispricing %"]], axis=-1)
    world = go.Figure(
        go.Scattergeo(
            lat=df["lat"], lon=df["lon"], text=df["City"], mode="markers", customdata=custom,
            marker=dict(
                size=14, color=warming, cmin=-limit, cmax=limit,
                colorscale=[[0.0, p["cold"]], [0.5, p["neutral"]], [1.0, p["warm"]]],
                line=dict(width=1, color=p["muted"]),
                colorbar=dict(
                    title=dict(text="Season warming<br>°C per decade", font=dict(color=p["ink2"])),
                    thickness=12, len=0.75, tickfont=dict(color=p["muted"]),
                ),
            ),
            hovertemplate=(
                "<b>%{text}</b><br>Season warming %{marker.color:+.2f} °C/decade"
                f"<br>{period.index} trend %{{customdata[0]:+.1f}}% per decade (q = %{{customdata[1]:.3f}})"
                "<br>Naive mispricing %{customdata[2]:+.0f}%<extra></extra>"
            ),
        )
    )
    world.update_geos(
        projection_type="natural earth", showland=True, landcolor=p["land"], showcountries=True,
        countrycolor=p["axis"], coastlinecolor=p["axis"], showframe=False, bgcolor="rgba(0,0,0,0)",
        fitbounds="locations",
    )
    world.update_layout(
        height=430, margin=dict(l=0, r=0, t=8, b=0), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=p["ink2"]), hoverlabel=dict(font=dict(family=FONT)),
    )

    ranked = df.sort_values("Mispricing %")
    bars = go.Figure()
    for flag, name, colour in ((True, "Significant trend (FDR 5%)", p["lookback"]), (False, "Not significant", p["history"])):
        part = ranked[ranked["Significant"] == flag]
        if part.empty:
            continue
        bars.add_bar(
            y=part["City"], x=part["Mispricing %"], orientation="h", name=name, marker_color=colour,
            customdata=np.stack([part["Naive"], part["Trend-adjusted"], part["Currency"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Naive vs trend-adjusted %{x:+.0f}%"
            "<br>Naive %{customdata[2]}%{customdata[0]:,.0f} · adjusted %{customdata[2]}%{customdata[1]:,.0f}<extra></extra>",
        )
    style(bars, height=90 + 26 * len(ranked), x_title="Naive vs adjusted premium", x_grid=True)
    bars.update_layout(barmode="relative", bargap=0.35, barcornerradius=4)
    bars.update_yaxes(categoryorder="array", categoryarray=ranked["City"].tolist(), showgrid=False)
    bars.update_xaxes(zeroline=True, zerolinecolor=p["axis"], zerolinewidth=1, ticksuffix="%")
    return world, bars


def render_overview(period: ContractPeriod, pricing: PricingInputs, pricing_t: tuple, model_t: tuple) -> None:
    cached = [city for city in CITIES if power_client.is_cached(city.key)]
    missing = [city for city in CITIES if not power_client.is_cached(city.key)]
    if missing:
        with st.container(border=True):
            st.markdown(
                f"**{len(missing)} of {len(CITIES)} cities have no data yet.** The app sends one request per "
                "city to the NASA POWER API (usually 5–30 s each) and caches the result in `data/cache/`."
            )
            if st.button(f"Download NASA POWER data for {len(missing)} cities", type="primary", key="download_all"):
                errors = download(missing)
                if errors:
                    st.error("Some downloads failed:\n\n" + "\n".join(f"- {e}" for e in errors))
                else:
                    st.rerun()
    if not cached:
        st.info("Download the data above, or pick one city in the sidebar and open its case file.")
        return

    rows, failures = [], []
    with st.spinner("Pricing every city…"):
        for city in cached:
            try:
                rows.append(summary_row(run_analysis(astuple(city), period.key, pricing_t, model_t, data_version(city.key))))
            except Exception as exc:
                failures.append(f"{city.label}: {exc}")
    if failures:
        st.warning("Skipped:\n\n" + "\n".join(f"- {f}" for f in failures))
    if not rows:
        return

    df = pd.DataFrame(rows)
    df["q_value"] = benjamini_hochberg(df["p_value"].to_numpy())
    df["Significant"] = df["q_value"] < 0.05
    n_sig = int(df["Significant"].sum())
    median_gap = float(np.nanmedian(df["Mispricing %"]))
    strike_words = "at-the-money" if pricing.strike_offset == 0 else f"{pricing.strike_offset:+.0%} strike"

    tiles = st.columns(4)
    tiles[0].metric("Cities with a significant trend", f"{n_sig} of {len(df)}",
                    help="Mann–Kendall test (Hamed–Rao correction) on the seasonal index, Benjamini–Hochberg FDR 5%.")
    tiles[1].metric("Median season warming per decade", f"{np.nanmedian(df['Warming °C / decade']):+.2f} °C")
    tiles[2].metric("Median naive mispricing", pct(median_gap),
                    help="Naive burn price vs trend-adjusted burn price for the same option.")
    tiles[3].metric("Season being priced", str(df["Target season"].mode().iloc[0]))
    st.markdown(
        f"**{n_sig} of {len(df)}** cities show a statistically significant trend in the **{period.label}** index "
        f"after controlling the false-discovery rate. A pricer who averages the last **{pricing.lookback} seasons** "
        f"would quote the {strike_words} {pricing.kind} a median **{pct(median_gap)}** away from the "
        "trend-adjusted price."
    )

    world, bars = overview_figures(df, period, pricing)
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Where is it changing?** Warming of the contract season's average temperature")
        show(world)
    with right:
        st.markdown(f"**What does it do to prices?** {pricing.kind.title()} on {period.label}")
        show(bars)

    with st.expander("Table view (all cities)", expanded=False):
        columns = ["City", "Station", "Seasons", "Mean index", "Trend / decade", "Trend % / decade",
                   "Warming °C / decade", "p_value", "q_value", "Significant", "Unit", "Currency", "Strike",
                   "Naive", "Trend-adjusted", "Model", "Mispricing %", "Naive vs model %"]
        st.dataframe(
            df[columns], hide_index=True,
            column_config={
                "Mean index": st.column_config.NumberColumn(format="%.0f"),
                "Trend / decade": st.column_config.NumberColumn(format="%+.0f"),
                "Trend % / decade": st.column_config.NumberColumn(format="%+.1f%%"),
                "Warming °C / decade": st.column_config.NumberColumn(format="%+.2f"),
                "p_value": st.column_config.NumberColumn("p (MK-HR)", format="%.4f"),
                "q_value": st.column_config.NumberColumn("q (FDR)", format="%.4f"),
                "Strike": st.column_config.NumberColumn(format="%.0f"),
                "Naive": st.column_config.NumberColumn(format="%.0f"),
                "Trend-adjusted": st.column_config.NumberColumn(format="%.0f"),
                "Model": st.column_config.NumberColumn(format="%.0f"),
                "Mispricing %": st.column_config.NumberColumn(format="%+.0f%%"),
                "Naive vs model %": st.column_config.NumberColumn(format="%+.0f%%"),
            },
        )
        st.download_button(
            "Download CSV", df[columns].to_csv(index=False).encode(), file_name=f"mispriced_winters_{period.key}.csv",
            mime="text/csv", key="download_overview_csv",
        )


def history_figure(a: CityAnalysis) -> go.Figure:
    p = palette()
    hist = a.index_history
    seasons = hist.index.to_numpy()
    values = hist.to_numpy()
    labels = np.array([a.period.season_label(int(s)) for s in seasons])
    in_lookback = np.isin(seasons, a.lookback_seasons)
    fig = go.Figure()
    fig.add_scatter(
        x=seasons, y=values, mode="lines+markers", name="Season index",
        line=dict(color=p["history"], width=2), marker=dict(size=8, color=p["history"]),
        customdata=labels, hovertemplate="%{customdata}: %{y:,.0f}<extra>Season index</extra>",
    )
    fig.add_scatter(
        x=seasons[in_lookback], y=values[in_lookback], mode="markers", name=f"Last {in_lookback.sum()} seasons (naive burn)",
        marker=dict(size=9, color=p["lookback"], line=dict(width=2, color=p["surface"])),
        customdata=labels[in_lookback], hovertemplate="%{customdata}: %{y:,.0f}<extra>Lookback</extra>",
    )
    xs = np.array([seasons.min(), a.target_season], dtype=float)
    fig.add_scatter(
        x=xs, y=a.index_trend.intercept + a.index_trend.slope * xs, mode="lines", name="Sen trend",
        line=dict(color=p["adjusted"], width=2), hoverinfo="skip",
    )
    naive = a.prices["naive"].expected_index
    fig.add_scatter(
        x=[a.lookback_seasons.min(), a.target_season], y=[naive, naive], mode="lines", name="Naive expectation",
        line=dict(color=p["naive"], width=2), hovertemplate="Naive expectation %{y:,.0f}<extra></extra>",
    )
    for key, name, symbol, dodge in (("naive", "Naive", "circle", -0.45), ("detrended", "Trend-adjusted", "circle", 0.0),
                                     ("model", "Model", "diamond", 0.45)):
        colour = p["adjusted"] if key == "detrended" else p[key]
        fig.add_scatter(
            x=[a.target_season + dodge], y=[a.prices[key].expected_index], mode="markers",
            name=f"{name} expectation for {a.target_label}", showlegend=key != "naive",
            marker=dict(size=12, color=colour, symbol=symbol, line=dict(width=2, color=p["surface"])),
            hovertemplate=f"{name} expectation for {a.target_label}: %{{y:,.0f}}<extra></extra>",
        )
    fig.add_hline(
        y=a.strike, line=dict(color=p["muted"], width=1, dash="dot"),
        annotation_text=f"strike {a.strike:,.0f}", annotation_position="bottom left",
        annotation_font=dict(color=p["muted"], size=11),
    )
    x_title = "Season (year it starts)" if a.period.wraps_year else "Year"
    style(fig, height=420, x_title=x_title, y_title=f"{a.period.index} index ({a.city.unit_label} degree days)")
    fig.update_layout(hovermode="closest")
    return fig


def distribution_figure(a: CityAnalysis) -> go.Figure:
    p = palette()
    rows = [("Naive burn", a.naive_index, p["naive"]), ("Trend-adjusted burn", a.detrended_index, p["adjusted"])]
    fig = go.Figure()
    fig.add_trace(
        go.Violin(
            x=a.simulated_index, y0="Monte Carlo model", orientation="h", name="Monte Carlo model",
            line=dict(color=p["model"], width=2), fillcolor=rgba(p["model"], 0.12), points=False,
            meanline_visible=True, hoverinfo="skip", showlegend=False, spanmode="hard",
        )
    )
    for name, values, colour in rows:
        fig.add_trace(
            go.Box(
                x=values, y0=name, orientation="h", name=name, boxpoints="all", jitter=0.5, pointpos=0,
                marker=dict(size=8, color=colour, line=dict(width=1, color=p["surface"])),
                line=dict(color=colour, width=2), fillcolor="rgba(0,0,0,0)", boxmean=True, showlegend=False,
                hovertemplate=f"{name}: %{{x:,.0f}}<extra></extra>",
            )
        )
    observed = np.concatenate([a.naive_index, a.detrended_index])
    sim_lo, sim_hi = np.percentile(a.simulated_index, [0.5, 99.5])
    lo, hi = min(sim_lo, observed.min(), a.strike), max(sim_hi, observed.max(), a.strike)
    pad = 0.05 * (hi - lo)
    x0, x1 = (a.strike, hi + pad) if a.spec.kind == "call" else (lo - pad, a.strike)
    fig.add_vrect(x0=x0, x1=x1, fillcolor=p["muted"], opacity=0.08, line_width=0,
                  annotation_text=f"{a.spec.kind} pays", annotation_position="top right" if a.spec.kind == "call" else "top left",
                  annotation_font=dict(color=p["muted"], size=11))
    fig.add_vline(x=a.strike, line=dict(color=p["ink2"], width=1, dash="dot"))
    style(fig, height=340, x_title=f"{a.period.index} index for {a.target_label} ({a.city.unit_label} degree days)", x_grid=True)
    fig.update_xaxes(range=[lo - pad, hi + pad])
    fig.update_yaxes(categoryorder="array", categoryarray=["Monte Carlo model", "Trend-adjusted burn", "Naive burn"], showgrid=False)
    return fig


def lookback_figure(a: CityAnalysis, chosen: int) -> go.Figure:
    p = palette()
    table = lookback_sensitivity(a)
    cur = a.city.currency
    fig = go.Figure()
    fig.add_scatter(x=table["lookback"], y=table["naive_price"], mode="lines+markers", name="Naive burn",
                    line=dict(color=p["naive"], width=2), marker=dict(size=8),
                    hovertemplate=f"Naive burn {cur}%{{y:,.0f}}<extra></extra>")
    fig.add_scatter(x=table["lookback"], y=table["detrended_price"], mode="lines+markers", name="Trend-adjusted burn",
                    line=dict(color=p["adjusted"], width=2), marker=dict(size=8),
                    hovertemplate=f"Trend-adjusted {cur}%{{y:,.0f}}<extra></extra>")
    fig.add_scatter(x=[table["lookback"].min(), table["lookback"].max()], y=[a.prices["model"].price] * 2, mode="lines",
                    name="Monte Carlo model", line=dict(color=p["model"], width=2),
                    hovertemplate=f"Model {cur}%{{y:,.0f}}<extra></extra>")
    fig.add_vline(x=chosen, line=dict(color=p["muted"], width=1, dash="dot"),
                  annotation_text="your lookback", annotation_font=dict(color=p["muted"], size=11))
    style(fig, height=340, x_title="Lookback (seasons)", y_title=f"Premium ({cur})")
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero")
    return fig


def climate_figure(a: CityAnalysis) -> go.Figure:
    p = palette()
    model = a.model
    first_year = model.fit_start.year + (1 if model.fit_start.dayofyear > 1 else 0)
    target_start = pd.Timestamp(year=a.target_season, month=a.period.months[0], day=1)
    years = [(first_year, p["history"]), (target_start.year, p["adjusted"])]
    axis_dates = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    fig = go.Figure()
    for year, colour in years:
        dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        mean = model.mean(dates)[:365]
        fig.add_scatter(x=axis_dates, y=mean, mode="lines", name=f"{year} climate (model mean)",
                        line=dict(color=colour, width=2), hovertemplate="%{x|%d %b}: %{y:.1f} °C<extra>" + str(year) + "</extra>")
    style(fig, height=300, y_title="Daily mean temperature (°C)")
    fig.update_xaxes(tickformat="%b", dtick="M1")
    fig.update_layout(hovermode="x unified")
    return fig


def sigma_figure(a: CityAnalysis) -> go.Figure:
    p = palette()
    months = pd.date_range("2001-01-01", periods=12, freq="MS").strftime("%b")
    fig = go.Figure(go.Bar(x=months, y=a.model.sigma[1:], marker_color=p["ink2"],
                           hovertemplate="%{x}: σ = %{y:.2f} °C<extra></extra>"))
    style(fig, height=300, y_title="Daily shock σ (°C)")
    fig.update_layout(bargap=0.45, barcornerradius=4, showlegend=False)
    return fig


def render_city(a: CityAnalysis, pricing: PricingInputs) -> None:
    city, period = a.city, a.period
    cur, unit = city.currency, city.unit_label
    hist = a.index_history
    first, last = period.season_label(int(hist.index.min())), period.season_label(int(hist.index.max()))

    st.subheader(f"Case file: {city.label} · {period.label}")
    st.caption(
        f"{city.station} · NASA POWER grid cell at {city.lat:.2f}°, {city.lon:.2f}° · "
        f"{len(hist)} complete seasons ({first} to {last}) · pricing the {a.target_label} season"
    )

    base = f"{city.base_temp:g}{unit}"
    formula = f"max(0, {base} − Tavg)" if period.index == "HDD" else f"max(0, Tavg − {base})"
    significant = a.index_trend.significant()
    cards = st.columns(4)
    with cards[0].container(border=True):
        st.markdown("**What is changing?**")
        st.markdown(f"#### {period.demand} degree days")
        st.caption(f"{period.label}: daily {formula} summed over the season")
    with cards[1].container(border=True):
        st.markdown("**Where?**")
        st.markdown(f"#### {city.name}")
        st.caption(f"{city.station} · {city.lat:.2f}°, {city.lon:.2f}°")
    with cards[2].container(border=True):
        st.markdown("**How much?**")
        st.markdown(f"#### {a.trend_per_decade:+,.0f} {period.index} / decade")
        st.caption(f"{a.trend_pct_per_decade:+.1f}% of an average season · season temperature "
                   f"{a.warming_c_per_decade:+.2f} °C per decade")
    with cards[3].container(border=True):
        st.markdown("**Is it significant?**")
        st.markdown("#### Yes" if significant else "#### Not at 5%")
        st.caption(f"Mann–Kendall with Hamed–Rao correction: p = {a.index_trend.p_value:.3g}, "
                   f"Kendall τ = {a.index_trend.tau:+.2f}")

    naive, adjusted, model = (a.prices[k] for k in ("naive", "detrended", "model"))
    word = "overprices" if a.mispricing > 0 else "underprices"
    with st.container(border=True):
        st.markdown(
            f"**Verdict.** For the {a.target_label} {period.index} {a.spec.kind} (strike {a.strike:,.0f}, "
            f"{cur}{a.spec.tick:g} per {period.index}), a pricer who averages the last {len(a.lookback_seasons)} seasons "
            f"**{word} it by {money(abs(a.mispricing), cur)} ({pct(a.mispricing_pct)})** compared with the same seasons "
            f"adjusted for the trend. On a linear {cur}{a.spec.tick:g}/{period.index} future the same bias is worth "
            f"**{money(a.futures_mispricing, cur)}**. The Monte Carlo model, built independently from daily data, "
            f"says {money(model.price, cur)}."
        )
    def vs_naive(value: float) -> str:
        return f"{pct((value / naive.price - 1.0) * 100.0) if naive.price > 0 else 'n/a'} vs naive"

    tiles = st.columns(3)
    tiles[0].metric("Naive burn", money(naive.price, cur), help=f"Average payout over the last {naive.n} seasons, as they happened.")
    tiles[1].metric("Trend-adjusted burn", money(adjusted.price, cur), delta=vs_naive(adjusted.price), delta_color="off",
                    help="The same seasons with every day shifted to the target season's climate.")
    tiles[2].metric("Monte Carlo model", money(model.price, cur), delta=vs_naive(model.price), delta_color="off",
                    help=f"{model.n:,} simulated seasons from the seasonal AR({a.model.ar_order}) model.")

    st.markdown(f"**The settlement index, season by season**, and three expectations for {a.target_label}")
    show(history_figure(a))
    left, right = st.columns(2)
    with left:
        st.markdown("**Who expects what?** Index outcomes behind each price")
        show(distribution_figure(a))
    with right:
        st.markdown("**The longer the memory, the bigger the error** — premium vs lookback")
        show(lookback_figure(a, pricing.lookback))

    with st.expander("Model diagnostics"):
        m = a.model
        target_dates = pd.date_range(pd.Timestamp(year=a.target_season, month=period.months[0], day=1), periods=28, freq="D")
        st.markdown(
            f"- Fitted on {m.n_obs:,} days ({m.fit_start:%Y-%m-%d} to {m.fit_end:%Y-%m-%d}); the mean function explains "
            f"R² = {m.r2:.3f} of daily variance.\n"
            f"- AR({m.ar_order}) coefficients {np.array2string(m.phi, precision=3)}; persistence Σφ = {m.persistence:.3f}, "
            f"half-life ≈ {m.half_life_days:.1f} days.\n"
            f"- Model warming of the mean at the start of the season: {m.warming_rate(target_dates):+.2f} °C per decade "
            f"(Sen slope of the season average: {a.warming_c_per_decade:+.2f} °C per decade).\n"
            f"- Simulated index std {a.raw_simulated_std:,.0f} vs detrended historical std {a.historical_detrended_std:,.0f} "
            f"{period.index}"
            + (" — rescaled to match history." if not np.isclose(np.std(a.simulated_index, ddof=1), a.raw_simulated_std, rtol=1e-6) else ".")
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Climate then and now** (model mean)")
            show(climate_figure(a))
        with c2:
            st.markdown("**Day-to-day volatility by month**")
            show(sigma_figure(a))

    with st.expander("Table view"):
        prices = pd.DataFrame(
            [{"Method": r.method, "Expected index": r.expected_index, "Index std": r.index_std,
              f"Price ({cur})": r.price, "± std error": r.stderr, "Scenarios": r.n} for r in a.prices.values()]
        )
        st.dataframe(prices, hide_index=True, column_config={
            "Expected index": st.column_config.NumberColumn(format="%.1f"),
            "Index std": st.column_config.NumberColumn(format="%.1f"),
            f"Price ({cur})": st.column_config.NumberColumn(format="%.0f"),
            "± std error": st.column_config.NumberColumn(format="%.0f"),
        })
        seasons = pd.DataFrame({
            "Season": [period.season_label(int(s)) for s in hist.index],
            f"{period.index} index": hist.to_numpy(),
            f"Moved to {a.target_label} climate": a.detrended_history.reindex(hist.index).to_numpy(),
            f"Season mean temperature ({unit})": a.mean_temp_history.reindex(hist.index).to_numpy(),
            "In naive lookback": np.isin(hist.index.to_numpy(), a.lookback_seasons),
        })
        st.dataframe(seasons, hide_index=True, column_config={
            f"{period.index} index": st.column_config.NumberColumn(format="%.1f"),
            f"Moved to {a.target_label} climate": st.column_config.NumberColumn(format="%.1f"),
            f"Season mean temperature ({unit})": st.column_config.NumberColumn(format="%.2f"),
        })


def render_city_tab(city: City, period: ContractPeriod, pricing: PricingInputs, pricing_t: tuple, model_t: tuple) -> None:
    if not power_client.is_cached(city.key):
        st.info(f"No NASA POWER data for **{city.label}** yet.")
        if st.button(f"Download daily temperatures for {city.label}", type="primary", key="download_city"):
            with st.spinner("Requesting 1981 to today from NASA POWER…"):
                try:
                    power_client.get_daily(city.lat, city.lon, city.key, refresh=True)
                except Exception as exc:
                    st.error(f"Download failed: {exc}")
                    return
            st.cache_data.clear()
            st.rerun()
        return
    try:
        analysis = run_analysis(astuple(city), period.key, pricing_t, model_t, data_version(city.key))
    except Exception as exc:
        st.error(f"Could not analyse {city.label}: {exc}")
        return
    render_city(analysis, pricing)


def backtest_figures(bt: BacktestResult, city: City, period: ContractPeriod) -> tuple[go.Figure, go.Figure]:
    p = palette()
    t = bt.table
    x = t.index.to_numpy()
    labels = t["label"].to_numpy()
    forecast = go.Figure()
    forecast.add_scatter(x=x, y=t["realized_index"], mode="lines+markers", name="What happened",
                         line=dict(color=p["ink2"], width=2), marker=dict(size=8), customdata=labels,
                         hovertemplate="%{customdata}: %{y:,.0f}<extra>Realised</extra>")
    forecast.add_scatter(x=x, y=t["naive_forecast"], mode="lines+markers", name="Naive forecast",
                         line=dict(color=p["naive"], width=2), marker=dict(size=8), customdata=labels,
                         hovertemplate="%{customdata}: %{y:,.0f}<extra>Naive</extra>")
    forecast.add_scatter(x=x, y=t["detrended_forecast"], mode="lines+markers", name="Trend-adjusted forecast",
                         line=dict(color=p["adjusted"], width=2), marker=dict(size=8), customdata=labels,
                         hovertemplate="%{customdata}: %{y:,.0f}<extra>Trend-adjusted</extra>")
    style(forecast, height=360, x_title="Season priced", y_title=f"{period.index} index ({city.unit_label} degree days)")
    forecast.update_layout(hovermode="x unified")

    pnl = go.Figure()
    pnl.add_scatter(x=x, y=t["naive_pnl"].cumsum(), mode="lines+markers", name="Bought at the naive price",
                    line=dict(color=p["naive"], width=2), marker=dict(size=8), customdata=labels,
                    hovertemplate=f"%{{customdata}}: {city.currency}%{{y:,.0f}}<extra>Naive</extra>")
    pnl.add_scatter(x=x, y=t["detrended_pnl"].cumsum(), mode="lines+markers", name="Bought at the trend-adjusted price",
                    line=dict(color=p["adjusted"], width=2), marker=dict(size=8), customdata=labels,
                    hovertemplate=f"%{{customdata}}: {city.currency}%{{y:,.0f}}<extra>Trend-adjusted</extra>")
    style(pnl, height=360, x_title="Season priced", y_title=f"Cumulative buyer P&L ({city.currency})")
    pnl.update_yaxes(zeroline=True, zerolinecolor=p["axis"], zerolinewidth=1)
    pnl.update_layout(hovermode="x unified")
    return forecast, pnl


def render_backtest_tab(city: City, period: ContractPeriod, pricing: PricingInputs, pricing_t: tuple) -> None:
    if not power_client.is_cached(city.key):
        st.info(f"Download data for **{city.label}** in the City case file tab first.")
        return
    try:
        bt = run_backtest(astuple(city), period.key, pricing_t, data_version(city.key))
    except Exception as exc:
        st.error(f"Backtest failed: {exc}")
        return
    s = bt.summary
    if not s.get("n"):
        st.info("Not enough seasons for a walk-forward test with this lookback — try a shorter one.")
        return
    t = bt.table
    cur = city.currency
    st.subheader(f"Walk-forward backtest: {city.label} · {period.label}")
    st.markdown(
        f"Each season from **{t['label'].iloc[0]}** to **{t['label'].iloc[-1]}** is priced using **only the seasons "
        f"before it**: the naive pricer averages the previous {pricing.lookback}; the trend-adjusted pricer moves the "
        "same seasons to that year's climate with a trend estimated on past data only. Then we compare with what happened."
    )
    tiles = st.columns(5)
    tiles[0].metric("Seasons tested", f"{s['n']}")
    tiles[1].metric("Naive bias", f"{s['naive_index_bias']:+,.0f} {period.index}",
                    help=f"Average of forecast − realised index. t = {s['naive_index_bias_t']:.2f}")
    tiles[2].metric("Trend-adjusted bias", f"{s['detrended_index_bias']:+,.0f} {period.index}",
                    help=f"Average of forecast − realised index. t = {s['detrended_index_bias_t']:.2f}")
    tiles[3].metric("P&L buying at naive", money(s["naive_buyer_pnl"], cur),
                    help="Sum of payout − premium from buying one option every season at the naive price.")
    tiles[4].metric("P&L buying at adjusted", money(s["detrended_buyer_pnl"], cur),
                    help="Sum of payout − premium from buying one option every season at the trend-adjusted price.")
    t_stat = s["naive_index_bias_t"]
    verdict = (
        f"The naive forecast is **systematically biased** (t = {t_stat:.1f}): it missed by {s['naive_index_bias']:+,.0f} "
        f"{period.index} on average, against {s['detrended_index_bias']:+,.0f} for the trend-adjusted forecast."
        if np.isfinite(t_stat) and abs(t_stat) >= 2
        else f"With {s['n']} seasons the naive bias ({s['naive_index_bias']:+,.0f} {period.index}, t = {t_stat:.1f}) "
        "is not statistically significant on its own — weather noise is large relative to the trend in a single city."
    )
    st.markdown(verdict)
    forecast, pnl = backtest_figures(bt, city, period)
    left, right = st.columns(2)
    with left:
        st.markdown("**Forecast vs reality** — expected index when the contract was priced")
        show(forecast)
    with right:
        st.markdown("**Who loses money?** Buying one option every season")
        show(pnl)
    with st.expander("Table view"):
        st.dataframe(t.reset_index(drop=True), hide_index=True)


def render_method() -> None:
    st.markdown(
        """
### The problem
Utilities, energy traders and farmers hedge weather with contracts that pay out on **degree days**.
A *heating degree day* (HDD) counts how far a day's average temperature falls below a base (65 °F in US
contracts, 18 °C in Europe). Many desks still price these contracts with **burn analysis**: replay the last
10–30 seasons and average the payouts. If winters are getting warmer, the past contains too many cold
seasons — so naive burn analysis **overprices HDD calls and underprices CDD calls**.

### How the app answers the challenge's four questions
| Question | Answer in the app |
|---|---|
| What is changing? | The seasonal HDD/CDD settlement index (Nov–Mar, May–Sep or single months) |
| Where is it changing? | Grid cells at 8 US settlement airports, 5 European airports and 5 Ukrainian cities, plus any custom point |
| How much? | Theil–Sen slope per decade, in degree days and in % of an average season |
| Is it significant? | Mann–Kendall test with Hamed–Rao autocorrelation correction; Benjamini–Hochberg FDR across cities |

### Three prices for the same option
"""
    )
    st.latex(r"I_s=\sum_{d\in s}\max(0,\;B-T_d),\qquad T_d=\tfrac12\,(T^{\max}_d+T^{\min}_d),\qquad \text{payoff}=\text{tick}\cdot\max(I-K,\,0)")
    st.markdown("**1 · Naive burn** — the last *L* seasons, as they happened:")
    st.latex(r"P_{\text{naive}}=e^{-r\tau}\,\frac1L\sum_{s=S-L}^{S-1}\text{payoff}(I_s)")
    st.markdown(
        "**2 · Trend-adjusted burn** — the same seasons, every day shifted to the target season's climate with the "
        "Sen slope β̂ of the season's mean temperature, then the degree days are recomputed (so the truncation at the "
        "base stays exact):"
    )
    st.latex(r"T_d^{\,\to S}=T_d+\hat\beta\,(S-s),\qquad P_{\text{adj}}=e^{-r\tau}\,\frac1L\sum_{s}\text{payoff}\big(I_s^{\,\to S}\big)")
    st.markdown(
        "**3 · Seasonal AR Monte Carlo** — a daily model in the spirit of Alaton, Djehiche & Stillberger (2002): "
        "deterministic mean with trend and harmonics (the trend may differ by season), AR(p) anomalies with a "
        "separate volatility for every month, shocks bootstrapped from history. Daily models tend to understate "
        "the variance of seasonal sums, so the simulated index is optionally rescaled to the detrended historical "
        "variance around its trend-aware mean."
    )
    st.latex(r"T_t=a+b\,t+\sum_{k=1}^{3}\big[c_k\cos k\omega t+d_k\sin k\omega t\big]+t\,(e\cos\omega t+f\sin\omega t)+X_t,\qquad X_t=\sum_{i=1}^{p}\phi_iX_{t-i}+\sigma_{m(t)}\,\varepsilon_t")
    st.markdown(
        """
The **walk-forward backtest** prices every past season using only the seasons before it, so its verdict
contains no look-ahead.

### Caveats (read before trading)
- **Basis risk.** NASA POWER is a gridded reanalysis (MERRA-2 based, roughly 50 km cells); exchange contracts
  settle on one station's thermometer. Airport micro-climates and urban heat islands differ from the grid cell.
- **Day definition.** POWER's daily max/min may not use the station's local climatological day.
- **Data homogeneity.** Changes in the observing system can leave artificial jumps in reanalysis; check the
  findings against station records (for example NOAA GHCN-Daily) before relying on them.
- **Actuarial prices.** Prices are expected payoffs under the physical measure, with no risk loading or
  market price of risk; real quotes add a margin.
- **Linear trend, no forecasts.** No ENSO/NAO conditioning and no information from inside the season.
- Contracts for Ukrainian cities (and some European ones) are hypothetical.

### References
- Alaton, Djehiche & Stillberger (2002). *On modelling and pricing weather derivatives.* Applied Mathematical Finance 9(1).
- Benth & Šaltytė Benth (2007). *The volatility of temperature and pricing of weather derivatives.* Quantitative Finance 7(5).
- Jewson & Brix (2005). *Weather Derivative Valuation.* Cambridge University Press.
- Mann (1945); Kendall (1975); Sen (1968), JASA 63; Hamed & Rao (1998), Journal of Hydrology 204.
- Benjamini & Hochberg (1995). Journal of the Royal Statistical Society B 57.
- NASA POWER: https://power.larc.nasa.gov
"""
    )


city, period, pricing, model_inputs = sidebar()
pricing_t, model_t = astuple(pricing), astuple(model_inputs)

st.title("❄️ Mispriced Winters")
st.markdown(
    "Is a warming climate quietly mispricing weather derivatives? We test **four decades of NASA POWER "
    "temperatures** for trends in heating and cooling degree days, then show what a pricer who ignores the "
    "trend gets wrong — and prove it with a walk-forward backtest."
)
with st.expander("New to weather derivatives? 30-second version"):
    st.markdown(
        "A gas utility sells less gas in a mild winter. To protect itself it can buy an **HDD put** that pays when "
        "the winter's heating degree days come in low; an energy buyer worried about a cold winter buys an **HDD "
        "call**. The price of these options depends on what you expect the coming season to look like — and the "
        "naive answer, *“average the past 20 winters”*, ignores that those winters were colder than today's."
    )

tab_overview, tab_city, tab_backtest, tab_method = st.tabs(["🌍 Overview", "🔎 City case file", "📉 Backtest", "📚 Method"])
with tab_overview:
    render_overview(period, pricing, pricing_t, model_t)
with tab_city:
    render_city_tab(city, period, pricing, pricing_t, model_t)
with tab_backtest:
    render_backtest_tab(city, period, pricing, pricing_t)
with tab_method:
    render_method()

st.caption(ACKNOWLEDGEMENT + " · NASA Space Apps Challenge 2026 — Be an Earth System Trend Detective.")
