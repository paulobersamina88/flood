import math
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Metro Manila Flood Weather + Heat Index Dashboard", layout="wide")

DEFAULT_LOCATIONS = {
    "Manila": {"lat": 14.5995, "lon": 120.9842},
    "Valenzuela": {"lat": 14.7000, "lon": 120.9830},
    "Quezon City": {"lat": 14.6760, "lon": 121.0437},
    "Pasig": {"lat": 14.5764, "lon": 121.0851},
    "Marikina": {"lat": 14.6507, "lon": 121.1029},
}

WEATHER_API = "https://api.open-meteo.com/v1/forecast"


def compute_heat_index_c(temp_c: float, rh: float) -> float:
    """
    NOAA Rothfusz regression.
    Uses Fahrenheit internally, returns Celsius.
    For low temperatures, returns actual air temperature as fallback.
    """
    if pd.isna(temp_c) or pd.isna(rh):
        return np.nan

    if temp_c < 27:
        return float(temp_c)

    t_f = temp_c * 9 / 5 + 32

    hi_f = (
        -42.379
        + 2.04901523 * t_f
        + 10.14333127 * rh
        - 0.22475541 * t_f * rh
        - 0.00683783 * t_f * t_f
        - 0.05481717 * rh * rh
        + 0.00122874 * t_f * t_f * rh
        + 0.00085282 * t_f * rh * rh
        - 0.00000199 * t_f * t_f * rh * rh
    )

    if 80 <= t_f <= 112 and rh < 13:
        adjustment = ((13 - rh) / 4) * np.sqrt((17 - abs(t_f - 95)) / 17)
        hi_f -= adjustment
    elif 80 <= t_f <= 87 and rh > 85:
        adjustment = ((rh - 85) / 10) * ((87 - t_f) / 5)
        hi_f += adjustment

    return float((hi_f - 32) * 5 / 9)


def classify_heat_stress(heat_index_c: float) -> str:
    if pd.isna(heat_index_c):
        return "N/A"
    if heat_index_c >= 52:
        return "Extreme Danger"
    elif heat_index_c >= 41:
        return "Danger"
    elif heat_index_c >= 33:
        return "Extreme Caution"
    elif heat_index_c >= 27:
        return "Caution"
    return "Normal"


def heat_badge_html(level: str) -> str:
    styles = {
        "Normal": "#d1fae5|#065f46",
        "Caution": "#fef3c7|#92400e",
        "Extreme Caution": "#fed7aa|#9a3412",
        "Danger": "#fecaca|#991b1b",
        "Extreme Danger": "#e9d5ff|#6b21a8",
        "N/A": "#e5e7eb|#374151",
    }
    bg, fg = styles.get(level, "#e5e7eb|#374151").split("|")
    return f"""
    <div style="
        display:inline-block;
        padding:0.28rem 0.65rem;
        border-radius:999px;
        background:{bg};
        color:{fg};
        font-weight:700;
        font-size:0.85rem;
        text-align:center;
        min-width:130px;
    ">
        {level}
    </div>
    """


def compute_risk_score(row: pd.Series) -> tuple[int, str, str]:
    score = 0
    reasons = []

    precip = float(row.get("precipitation_mm", 0) or 0)
    rain = float(row.get("rain_mm", 0) or 0)
    rh = float(row.get("RH_2m_pct", 0) or 0)
    tpw = float(row.get("TPW_kg_m2", 0) or 0)
    wind = float(row.get("Wind_Speed_kmh", 0) or 0)

    if precip >= 20:
        score += 5
        reasons.append("very heavy precipitation")
    elif precip >= 10:
        score += 4
        reasons.append("heavy precipitation")
    elif precip >= 5:
        score += 3
        reasons.append("moderate precipitation")
    elif precip >= 1:
        score += 1
        reasons.append("light precipitation")

    if rain >= 15:
        score += 3
        reasons.append("heavy rain")
    elif rain >= 8:
        score += 2
        reasons.append("moderate rain")
    elif rain >= 2:
        score += 1
        reasons.append("light rain")

    if rh >= 95:
        score += 2
        reasons.append("very high humidity")
    elif rh >= 90:
        score += 1
        reasons.append("high humidity")

    if tpw >= 65:
        score += 3
        reasons.append("very high precipitable water")
    elif tpw >= 55:
        score += 2
        reasons.append("high precipitable water")
    elif tpw >= 45:
        score += 1
        reasons.append("elevated precipitable water")

    if wind >= 35:
        score += 1
        reasons.append("gusty winds")

    if score >= 9:
        level = "High"
    elif score >= 5:
        level = "Moderate"
    else:
        level = "Low"

    return score, level, ", ".join(reasons) if reasons else "No strong rainfall signal"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_open_meteo(lat: float, lon: float, tz: str, past_hours: int, future_hours: int) -> pd.DataFrame:
    past_days = max(1, math.ceil(past_hours / 24))
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join([
            "temperature_2m",
            "apparent_temperature",
            "relative_humidity_2m",
            "total_column_integrated_water_vapour",
            "wind_speed_10m",
            "precipitation",
            "rain",
        ]),
        "past_days": min(past_days, 92),
        "forecast_hours": future_hours,
        "timezone": tz,
    }
    resp = requests.get(WEATHER_API, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    hourly = data["hourly"]

    df = pd.DataFrame(
        {
            "time": pd.to_datetime(hourly["time"]),
            "Temperature_2m_C": hourly["temperature_2m"],
            "Apparent_Temperature_C": hourly["apparent_temperature"],
            "RH_2m_pct": hourly["relative_humidity_2m"],
            "TPW_kg_m2": hourly["total_column_integrated_water_vapour"],
            "Wind_Speed_kmh": hourly["wind_speed_10m"],
            "precipitation_mm": hourly["precipitation"],
            "rain_mm": hourly["rain"],
        }
    )
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()

    now = pd.Timestamp.now(tz=tz).tz_localize(None)
    start_hist = now - pd.Timedelta(hours=past_hours)
    end_fut = now + pd.Timedelta(hours=future_hours)
    df = df[(df.index >= start_hist) & (df.index <= end_fut)].copy()

    df["Heat_Index_C"] = df.apply(
        lambda r: compute_heat_index_c(r["Temperature_2m_C"], r["RH_2m_pct"]),
        axis=1,
    )
    df["Heat_Stress_Level"] = df["Heat_Index_C"].apply(classify_heat_stress)

    score_info = df.apply(compute_risk_score, axis=1, result_type="expand")
    score_info.columns = ["risk_score", "risk_level", "risk_reason"]
    df = pd.concat([df, score_info], axis=1)
    return df


def summary_metrics(df: pd.DataFrame) -> dict:
    now = pd.Timestamp.now().floor("h")
    future = df[df.index >= now]
    next_24 = future.head(24)

    if next_24.empty:
        return {
            "peak_risk": "N/A",
            "peak_precip": 0.0,
            "peak_tpw": 0.0,
            "peak_heat_index": 0.0,
            "hours_moderate_or_high": 0,
        }

    levels = next_24["risk_level"].tolist()
    peak_level = "Low"
    if "High" in levels:
        peak_level = "High"
    elif "Moderate" in levels:
        peak_level = "Moderate"

    return {
        "peak_risk": peak_level,
        "peak_precip": float(next_24["precipitation_mm"].max()),
        "peak_tpw": float(next_24["TPW_kg_m2"].max()),
        "peak_heat_index": float(next_24["Heat_Index_C"].max()),
        "hours_moderate_or_high": int(next_24["risk_level"].isin(["Moderate", "High"]).sum()),
    }


def make_download_table(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    for name, df in all_data.items():
        temp = df.reset_index().rename(columns={"index": "time"}).copy()
        temp.insert(0, "location", name)
        frames.append(temp)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def latest_snapshot(all_data: dict[str, pd.DataFrame], tz: str) -> pd.DataFrame:
    latest_rows = []
    current_hour = pd.Timestamp.now(tz=tz).tz_localize(None).floor("h")

    for name, df in all_data.items():
        future_df = df[df.index >= current_hour]
        row = future_df.iloc[0] if not future_df.empty else df.iloc[-1]
        row_time = future_df.index[0] if not future_df.empty else df.index[-1]

        latest_rows.append(
            {
                "location": name,
                "time": row_time,
                "temperature_C": row["Temperature_2m_C"],
                "apparent_temp_C": row["Apparent_Temperature_C"],
                "heat_index_C": row["Heat_Index_C"],
                "heat_stress": row["Heat_Stress_Level"],
                "risk_level": row["risk_level"],
                "risk_score": int(row["risk_score"]),
                "precipitation_mm": row["precipitation_mm"],
                "rain_mm": row["rain_mm"],
                "RH_2m_pct": row["RH_2m_pct"],
                "TPW_kg_m2": row["TPW_kg_m2"],
                "Wind_Speed_kmh": row["Wind_Speed_kmh"],
                "reason": row["risk_reason"],
            }
        )

    return pd.DataFrame(latest_rows).sort_values(
        ["heat_index_C", "risk_score", "precipitation_mm"],
        ascending=[False, False, False],
    )


st.title("Metro Manila Flood Weather + Heat Index Dashboard")
st.caption(
    "Uses Open-Meteo weather data to estimate weather-driven flood risk and heat stress. "
    "This is a screening tool and not a direct street water-level or passability measurement."
)

with st.sidebar:
    st.header("Settings")
    tz = st.text_input("Timezone", value="Asia/Manila")
    past_hours = st.slider("Past hours", min_value=24, max_value=168, value=96, step=24)
    future_hours = st.slider("Future hours", min_value=24, max_value=168, value=120, step=24)

    st.markdown("### Locations")
    default_selected = list(DEFAULT_LOCATIONS.keys())[:2]
    selected = st.multiselect(
        "Choose cities",
        options=list(DEFAULT_LOCATIONS.keys()),
        default=default_selected,
    )
    run = st.button("Run weather scan", type="primary", use_container_width=True)

    st.markdown("### Add custom location")
    custom_name = st.text_input("Custom name")
    c1, c2 = st.columns(2)
    custom_lat = c1.number_input("Lat", value=14.6500, format="%.4f")
    custom_lon = c2.number_input("Lon", value=120.9800, format="%.4f")
    add_custom = st.button("Add custom location", use_container_width=True)

if "custom_locations" not in st.session_state:
    st.session_state.custom_locations = {}

if add_custom and custom_name.strip():
    st.session_state.custom_locations[custom_name.strip()] = {
        "lat": custom_lat,
        "lon": custom_lon,
    }
    st.success(f"Added {custom_name.strip()}.")

locations = {**DEFAULT_LOCATIONS, **st.session_state.custom_locations}

if not selected and locations:
    selected = list(locations.keys())[:1]

if run:
    if not selected:
        st.warning("Select at least one location.")
        st.stop()

    all_data = {}
    errors = []

    with st.spinner("Fetching weather data..."):
        for name in selected:
            try:
                coords = locations[name]
                all_data[name] = fetch_open_meteo(
                    coords["lat"], coords["lon"], tz, past_hours, future_hours
                )
            except Exception as e:
                errors.append(f"{name}: {e}")

    if errors:
        for err in errors:
            st.error(err)

    if not all_data:
        st.stop()

    latest_df = latest_snapshot(all_data, tz)

    st.subheader("Heat stress summary")
    hottest = latest_df.iloc[0]
    top1, top2, top3, top4 = st.columns(4)
    top1.metric("Hottest location now", hottest["location"])
    top2.metric("Heat Index", f"{hottest['heat_index_C']:.1f} °C")
    top3.metric("Air Temperature", f"{hottest['temperature_C']:.1f} °C")
    top4.markdown(heat_badge_html(hottest["heat_stress"]), unsafe_allow_html=True)

    st.subheader("24-hour outlook")
    metric_cols = st.columns(len(all_data))
    for col, (name, df) in zip(metric_cols, all_data.items()):
        metrics = summary_metrics(df)
        with col:
            st.markdown(f"**{name}**")
            st.metric("Peak flood risk", metrics["peak_risk"])
            st.metric("Max precip / hr", f"{metrics['peak_precip']:.1f} mm")
            st.metric("Max TPW", f"{metrics['peak_tpw']:.1f}")
            st.metric("Peak heat index", f"{metrics['peak_heat_index']:.1f} °C")
            st.metric("Moderate/High flood hrs", metrics["hours_moderate_or_high"])

    st.subheader("Latest status")

    for _, row in latest_df.iterrows():
        c1, c2, c3, c4, c5 = st.columns([1.4, 1.1, 1.1, 1.1, 2.8])
        c1.markdown(f"**{row['location']}**  \n{row['time']}")
        c2.metric("Temp", f"{row['temperature_C']:.1f} °C")
        c3.metric("Heat Index", f"{row['heat_index_C']:.1f} °C")
        c4.markdown(heat_badge_html(row["heat_stress"]), unsafe_allow_html=True)
        c5.write(
            f"Flood risk: **{row['risk_level']}** | "
            f"Rain: **{row['precipitation_mm']:.1f} mm** | "
            f"RH: **{row['RH_2m_pct']:.0f}%** | "
            f"Reason: {row['reason']}"
        )
        st.divider()

    with st.expander("Show latest status table"):
        st.dataframe(latest_df, use_container_width=True, hide_index=True)

    st.subheader("Trend charts")
    tab1, tab2, tab3, tab4 = st.tabs(
        ["RH & TPW", "Wind & Rain", "Temperature & Heat Index", "Per-location tables"]
    )

    with tab1:
        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax2 = ax1.twinx()
        for name, df in all_data.items():
            ax1.plot(df.index, df["RH_2m_pct"], linewidth=2, label=f"RH - {name}")
            ax2.plot(df.index, df["TPW_kg_m2"], linewidth=2, linestyle="--", label=f"TPW - {name}")
        ax1.axhline(85, linestyle=":", linewidth=1)
        ax2.axhline(65, linestyle=":", linewidth=1)
        ax1.set_ylabel("RH [%]")
        ax2.set_ylabel("TPW [kg/m²]")
        ax1.set_xlabel(f"Time ({tz})")
        ax1.axvline(pd.Timestamp.now(tz=tz).tz_localize(None), linewidth=1, alpha=0.6)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with tab2:
        fig2, ax = plt.subplots(figsize=(12, 4))
        for name, df in all_data.items():
            ax.plot(df.index, df["Wind_Speed_kmh"], linewidth=2, label=f"Wind - {name}")
            ax.plot(df.index, df["precipitation_mm"], linewidth=2, linestyle="--", label=f"Precip - {name}")
        ax.set_ylabel("km/h or mm")
        ax.set_xlabel(f"Time ({tz})")
        ax.axvline(pd.Timestamp.now(tz=tz).tz_localize(None), linewidth=1, alpha=0.6)
        ax.legend(loc="upper left")
        fig2.tight_layout()
        st.pyplot(fig2, use_container_width=True)

    with tab3:
        fig3, ax = plt.subplots(figsize=(12, 4))
        for name, df in all_data.items():
            ax.plot(df.index, df["Temperature_2m_C"], linewidth=2, label=f"Temp - {name}")
            ax.plot(df.index, df["Heat_Index_C"], linewidth=2, linestyle="--", label=f"Heat Index - {name}")
        ax.axhline(27, linestyle=":", linewidth=1)
        ax.axhline(33, linestyle=":", linewidth=1)
        ax.axhline(41, linestyle=":", linewidth=1)
        ax.axhline(52, linestyle=":", linewidth=1)
        ax.set_ylabel("°C")
        ax.set_xlabel(f"Time ({tz})")
        ax.axvline(pd.Timestamp.now(tz=tz).tz_localize(None), linewidth=1, alpha=0.6)
        ax.legend(loc="upper left")
        fig3.tight_layout()
        st.pyplot(fig3, use_container_width=True)

    with tab4:
        for name, df in all_data.items():
            st.markdown(f"**{name}**")
            display_df = df.reset_index().rename(columns={"index": "time"})
            st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Download results")
    download_df = make_download_table(all_data)
    csv_bytes = download_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        data=csv_bytes,
        file_name=f"metro_flood_weather_risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
    )

    st.info(
        "Interpretation note: this dashboard estimates possible flooding risk from weather conditions "
        "and heat stress from temperature and humidity. It does not prove that a specific street is "
        "passable or impassable. For street passability, you still need direct road/flood observations, "
        "LGU advisories, or water-level sensors."
    )
else:
    st.markdown(
        "Click **Run weather scan** to fetch the latest and recent weather signals for your selected Metro Manila locations."
    )
