from __future__ import annotations
import pandas as pd
import numpy as np

def preprocess_with_mapping(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    report = {"warnings": [], "dropped_rows": 0}

    dt_col = mapping.get("datetime")
    target_col = mapping.get("target")
    plant_col = mapping.get("plant_id")
    weather = mapping.get("weather", {}) or {}

    if not dt_col or dt_col not in df.columns:
        raise ValueError("datetime column not found in dataframe")
    if not target_col or target_col not in df.columns:
        raise ValueError("target column not found in dataframe")

    work = df.copy()
    work["ds_time"] = pd.to_datetime(work[dt_col], errors="coerce", utc=False)
    work["target"] = pd.to_numeric(work[target_col], errors="coerce")

    if plant_col and plant_col in work.columns:
        work["plant_id"] = work[plant_col].astype(str)
    else:
        work["plant_id"] = "default"

    def add_weather(new_name: str, col_name: str | None):
        if col_name and col_name in work.columns:
            work[new_name] = pd.to_numeric(work[col_name], errors="coerce")
        else:
            work[new_name] = np.nan

    add_weather("irradiance", weather.get("irradiance"))
    add_weather("ambient_temp", weather.get("ambient_temp"))
    add_weather("module_temp", weather.get("module_temp"))
    add_weather("wind_speed", weather.get("wind_speed"))
    add_weather("cloud_cover", weather.get("cloud_cover"))

    before = len(work)
    work = work.dropna(subset=["ds_time", "target"])
    after = len(work)
    report["dropped_rows"] = before - after

    work = work.sort_values("ds_time")

    work["hour"] = work["ds_time"].dt.hour
    work["dayofweek"] = work["ds_time"].dt.dayofweek
    work["month"] = work["ds_time"].dt.month

    for c in ["irradiance", "ambient_temp", "module_temp", "wind_speed", "cloud_cover"]:
        work[c] = work[c].ffill()
        med = work[c].median() if work[c].notna().any() else 0.0
        work[c] = work[c].fillna(med)

    return work, report