# Metro Manila Flood Weather Risk Dashboard

A Streamlit dashboard that uses Open-Meteo weather data to estimate weather-driven flood risk for Metro Manila locations.

## What it does
- pulls recent and forecast hourly weather data
- uses precipitation, rain, relative humidity, TPW, and wind speed
- computes a simple heuristic flood weather risk level: Low / Moderate / High
- compares multiple locations
- allows CSV export

## Important limitation
This app does **not** measure actual street flood depth or direct vehicle passability.
It is a weather-risk screening dashboard only.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
