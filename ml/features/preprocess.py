from __future__ import annotations
import pandas as pd
import numpy as np
import json 

def preprocess_with_mapping(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
    report = {"warnings": [], "dropped_rows": 0}

    work = df.copy()
    work.columns = [c.lower() for c in work.columns]

    dt_col = mapping.get("datetime", "").lower()
    target_col = mapping.get("target", "").lower()
    plant_col = mapping.get("plant_id", "").lower()
    weather = mapping.get("weather", {}) or {}

    if not dt_col or dt_col not in work.columns:
        raise ValueError(f"datetime column '{dt_col}' not found in dataframe")
    if not target_col or target_col not in work.columns:
        raise ValueError(f"target column '{target_col}' not found in dataframe")

    # يحول ال datetime ل Timestamp و ال target ل numeric و ال plant_id ل string 
    work["ds_time"] = pd.to_datetime(work[dt_col], errors="coerce", utc=False)
    work["target"] = pd.to_numeric(work[target_col], errors="coerce")

    if plant_col and plant_col in work.columns:
        work["plant_id"] = work[plant_col].astype(str)
    else:
        work["plant_id"] = "default"

    def add_weather(new_name: str, col_name: str | None):
        if col_name:
            col_lower = col_name.lower()
            if col_lower in work.columns:
                work[new_name] = pd.to_numeric(work[col_lower], errors="coerce")
            else:
                work[new_name] = np.nan
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

    # ---- Data Validation Layer ----
    validation_report = {
        "duplicate_timestamps": 0,
        "negative_target_values": 0,
        "night_hours_processed": 0,
        "outliers_detected": 0,
        "gaps_found": 0
    }

    before_dup = len(work)
    work = work.groupby("ds_time").mean(numeric_only=True).reset_index() 
    validation_report["duplicate_timestamps"] = int(before_dup - len(work))
    
    neg_mask = work["target"] < 0
    validation_report["negative_target_values"] = int(neg_mask.sum())
    work.loc[neg_mask, "target"] = 0

    night_mask = work["irradiance"] < 5
    work.loc[night_mask, "target"] = 0
    validation_report["night_hours_processed"] = int(night_mask.sum())

    Q1 = work["target"].quantile(0.25)
    Q3 = work["target"].quantile(0.75)
    IQR = Q3 - Q1
    outlier_limit = Q3 + 1.5 * IQR
    outlier_mask = work["target"] > outlier_limit
    validation_report["outliers_detected"] = int(outlier_mask.sum())
    work["target"] = work["target"].clip(upper=outlier_limit)

    # Missing Timestamps / Gaps
    time_diffs = work["ds_time"].diff().dt.total_seconds() / 3600
    validation_report["gaps_found"] = int((time_diffs > 1).sum())

    report_name = f"Validation_Report_{mapping.get('plant_id', 'default')}.json"
    try:
        from app.main import META_DIR 
        with open(META_DIR / report_name, "w") as f:
            json.dump(validation_report, f, indent=4)
    except:
        pass 
        
    work["hour"] = work["ds_time"].dt.hour
    work["dayofweek"] = work["ds_time"].dt.dayofweek
    work["month"] = work["ds_time"].dt.month
    work["hour_sin"] = np.sin(2 * np.pi * work["hour"] / 24)
    work["hour_cos"] = np.cos(2 * np.pi * work["hour"] / 24)

    # fill weather missing values by forward fill then median
    for c in ["irradiance", "ambient_temp", "module_temp", "wind_speed", "cloud_cover"]:
        work[c] = work[c].ffill()
        med = work[c].median() if work[c].notna().any() else 0.0
        work[c] = work[c].fillna(med)

    if "ac_power" in work.columns and "dc_power" in work.columns:
        work["inverter_efficiency"] = np.where(
            (work["dc_power"] > 0),
            work["ac_power"] / work["dc_power"],
            0.0
        )
    
    # [إصلاح لغم الـ Boolean] - تحويل النتيجة لـ 0 و 1 بدلاً من True/False لضمان عمل SVR/MLP
    if "irradiance" in work.columns and "target" in work.columns:
        work["possible_sensor_fault"] = ((work["irradiance"] > 50) & (work["target"] <= 0)).astype(int)
    else:
        work["possible_sensor_fault"] = 0


    final_features = ["target", "irradiance", "ambient_temp", "module_temp", 
                      "hour_sin", "hour_cos", "possible_sensor_fault"]
    
    existing_cols = [c for c in final_features if c in work.columns]
    

# ------- Feature Engineering -------
    work["day"] = work["ds_time"].dt.day
    work["dayofyear"] = work["ds_time"].dt.dayofyear
    work["weekday"] = work["ds_time"].dt.weekday
    work["weekofyear"] = work["ds_time"].dt.isocalendar().week.astype(int)

    work["season"] = work["month"] % 12 // 3 + 1
    work["daylight_indicator"] = (~night_mask).astype(int)

    # Lag features 
    work["lag_1"] = work["target"].shift(1)
    work["lag_24"] = work["target"].shift(24)
    work["lag_48"] = work["target"].shift(48)
    work["lag_168"] = work["target"].shift(168)

    work["rolling_mean_24"] = work["target"].rolling(window=24).mean()
    work["rolling_std_24"] = work["target"].rolling(window=24).std()

    
    work["year"] = work["ds_time"].dt.year
    
    
    work["month_sin"] = np.sin(2 * np.pi * work["month"] / 12)
    work["month_cos"] = np.cos(2 * np.pi * work["month"] / 12)


    work["rolling_mean_168"] = work["target"].rolling(window=168, min_periods=1).mean()
    
 
    work["efficiency_index"] = work["target"] / (work["irradiance"] + 0.1)

    final_features = [
        "target", "irradiance", "ambient_temp", "module_temp", 
        "hour_sin", "hour_cos", "month_sin", "month_cos",
        "year", "possible_sensor_fault", "day", "dayofyear", 
        "weekday", "weekofyear", "season", "daylight_indicator", 
        "lag_1", "lag_24", "lag_168", "rolling_mean_24", 
        "rolling_mean_168", "efficiency_index"
    ]
    
    existing_cols = [c for c in final_features if c in work.columns]
    return work[existing_cols], report