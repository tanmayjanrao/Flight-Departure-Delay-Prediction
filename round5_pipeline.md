# Flight Departure Delay Prediction — Round 5 Pipeline (Memory-Efficient)

Full pipeline: time-based split → airport code normalization → target/frequency/cyclical
encoding → weather features → engineered operational features (congestion, cascading delay
proxy, incoming aircraft delay) → **historical/lag features (route trailing delay,
flight-number rotation proxy, extended airport trend, holiday flag)** → model training
(Ridge, Linear Regression, XGBoost, LightGBM, CatBoost).

Optimized for Kaggle T4 x2: column-pruned loading, upfront downcasting, subset-only
groupby/sort operations instead of full-`df` operations, category-aligned merge keys instead
of `str` casts, and explicit `del` + `gc.collect()` once frames are no longer needed. No
feature definitions, causal boundaries, or leakage-safety logic were changed — only memory
footprint.

**Current best result (Round 5): XGBoost — RMSE 51.03, MAE 20.38, R² 0.17**

Progression:
- Round 1 (schedule only): R² ≈ 0.04
- Round 2 (+ weather): R² ≈ 0.04
- Round 3 (+ hourly congestion/cascading avg): R² ≈ 0.08
- Round 4 (+ incoming aircraft delay via merge_asof): R² ≈ 0.11
- **Round 5 (+ route trailing delay, flight-number lag, airport trend, holiday flag): R² ≈ 0.17**

---

## Cell 1 — Load Data & Time-Based Split

```python
# ============================================================================
# CELL 1 — LOAD DATA & TIME-BASED SPLIT
# ============================================================================

import pandas as pd
import numpy as np
import gc

FEATURES_CAT = ["airline", "flight_number", "origin_airport", "destination_airport"]
FEATURES_NUM = [
    "scheduled_flight_duration_minutes",
    "hour_of_day", "day_of_week", "month",
    "is_weekend", "is_redeye",
]
WEATHER_FEATURES = ["temperature_2m_c", "windspeed_10m_ms", "visibility_m", "cloudcover_low_pct"]
TARGET = "departure_delay_minutes"
TIME_COL = "scheduled_departure"

# flight_status / flight_type / cancellation_reason_available / has_departure_delay_target
# are never used anywhere in the pipeline -> skip loading them entirely instead of
# loading then dropping. Single biggest RAM saving here.
NEEDED_COLS = list(set(
    FEATURES_CAT + FEATURES_NUM + WEATHER_FEATURES +
    [TARGET, TIME_COL, "scheduled_arrival", "actual_arrival", "actual_departure",
     "route", "country"]
))

df = pd.read_parquet("your_file.parquet", columns=NEEDED_COLS)

df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"])
df["actual_arrival"] = pd.to_datetime(df["actual_arrival"])
df["actual_departure"] = pd.to_datetime(df["actual_departure"])

# downcast in place: float64->float32, int64->int32 (halves numeric footprint)
for c in df.select_dtypes(include="float64").columns:
    df[c] = df[c].astype("float32")
for c in df.select_dtypes(include="int64").columns:
    df[c] = df[c].astype("int32")

# repeated-value text columns -> category from the start
for c in ["airline", "flight_number", "origin_airport", "destination_airport", "route", "country"]:
    if c in df.columns:
        df[c] = df[c].astype("category")

train_mask = (df[TIME_COL] >= "2022-01-01") & (df[TIME_COL] <= "2024-12-31")
val_mask   = (df[TIME_COL] >= "2025-01-01") & (df[TIME_COL] <= "2025-12-31")
test_mask  = (df[TIME_COL] >= "2026-01-01") & (df[TIME_COL] <  "2026-06-02")

train_df = df.loc[train_mask].copy()
val_df   = df.loc[val_mask].copy()
test_df  = df.loc[test_mask].copy()
del train_mask, val_mask, test_mask
gc.collect()

trainval_df = pd.concat([train_df, val_df], axis=0)

print("train:", train_df.shape, " val:", val_df.shape, " test:", test_df.shape)
```

---

## Cell 2 — Normalize Airport Codes (ICAO → IATA)

```python
# ============================================================================
# CELL 2 — NORMALIZE AIRPORT CODES (ICAO → IATA)
# ============================================================================

airports = pd.read_csv("/kaggle/input/datasets/tjaycuz/airports/airports.csv", usecols=["icao_code", "iata_code"])
icao_to_iata = airports.dropna().set_index("icao_code")["iata_code"].to_dict()
del airports
gc.collect()

def normalize_airport(code):
    code = str(code).strip().upper()
    if len(code) == 3:
        return code
    if len(code) == 4:
        return icao_to_iata.get(code, code)
    return code

# normalize ONCE on df, then just re-slice into the splits instead of
# re-running .map() three more times over the same rows
for col in ["origin_airport", "destination_airport"]:
    df[col] = df[col].map(normalize_airport).astype("category")

for d in (train_df, val_df, test_df):
    for col in ["origin_airport", "destination_airport"]:
        d[col] = df.loc[d.index, col].astype("category")

trainval_df = pd.concat([train_df, val_df], axis=0)

for col in ["origin_airport", "destination_airport"]:
    print(f"\n{col}")
    print(train_df[col].astype(str).str.len().value_counts().sort_index())
```

---

## Cell 3 — Target Encoding (airline, origin_airport, destination_airport)

Leakage-safe: (a) K-fold out-of-fold within train, (b) val encoded from train-only stats,
(c) test encoded from train+val stats.

```python
# ============================================================================
# CELL 3 — TARGET ENCODING (AIRLINE, ORIGIN_AIRPORT, DESTINATION_AIRPORT)
# ============================================================================

def kfold_target_encode_train(train, col, target, n_splits=5, smoothing=10, seed=42):
    from sklearn.model_selection import KFold

    global_mean = train[target].mean()
    train_enc = pd.Series(index=train.index, dtype="float32")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr_idx, val_idx in kf.split(train):
        fold_train = train.iloc[tr_idx]
        stats = fold_train.groupby(col, observed=True)[target].agg(["mean", "count"])
        smooth_mean = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
        val_slice = train.iloc[val_idx]
        train_enc.iloc[val_idx] = val_slice[col].map(smooth_mean).fillna(global_mean).astype("float32")
        del fold_train, stats, smooth_mean, val_slice

    return train_enc.astype("float32")


def target_encode_from_reference(reference, apply_to, col, target, smoothing=10):
    global_mean = reference[target].mean()
    stats = reference.groupby(col, observed=True)[target].agg(["mean", "count"])
    smooth_mean = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return apply_to[col].map(smooth_mean).fillna(global_mean).astype("float32")


for col in ["airline", "origin_airport", "destination_airport"]:
    train_df[f"{col}_te"] = kfold_target_encode_train(train_df, col, TARGET, n_splits=5, smoothing=10)
    val_df[f"{col}_te"] = target_encode_from_reference(train_df, val_df, col, TARGET, smoothing=10)
    test_df[f"{col}_te"] = target_encode_from_reference(trainval_df, test_df, col, TARGET, smoothing=10)

gc.collect()
```

---

## Cell 4 — Frequency Encoding (flight_number)

```python
# ============================================================================
# CELL 4 — FREQUENCY ENCODING (FLIGHT_NUMBER)
# ============================================================================

freq_map_train = train_df["flight_number"].value_counts(normalize=True)
freq_map_trainval = trainval_df["flight_number"].value_counts(normalize=True)

train_df["flight_number_freq"] = train_df["flight_number"].map(freq_map_train).astype("float32")
val_df["flight_number_freq"] = val_df["flight_number"].map(freq_map_train).fillna(0).astype("float32")
test_df["flight_number_freq"] = test_df["flight_number"].map(freq_map_trainval).fillna(0).astype("float32")
del freq_map_train, freq_map_trainval
gc.collect()
```

---

## Cell 5 — Cyclical Encoding (hour, day_of_week, month)

```python
# ============================================================================
# CELL 5 — CYCLICAL ENCODING (HOUR, DAY_OF_WEEK, MONTH)
# ============================================================================

def add_cyclical(d, col, period):
    d[f"{col}_sin"] = np.sin(2 * np.pi * d[col] / period).astype("float32")
    d[f"{col}_cos"] = np.cos(2 * np.pi * d[col] / period).astype("float32")

for d in (train_df, val_df, test_df):
    add_cyclical(d, "hour_of_day", 24)
    add_cyclical(d, "day_of_week", 7)
    add_cyclical(d, "month", 12)
```

---

## Cell 6 — Feature Engineering: Airport Congestion + Cascading Delay Proxy

```python
# ============================================================================
# CELL 6 — FEATURE ENGINEERING: AIRPORT CONGESTION + CASCADING DELAY PROXY
# ============================================================================

cols_needed = ["airline", "origin_airport", "scheduled_departure",
               "actual_departure", "departure_delay_minutes"]
tmp = df[cols_needed].copy()

tmp["sched_hour_bucket"] = tmp["scheduled_departure"].dt.floor("h")
tmp["actual_hour_bucket"] = tmp["actual_departure"].dt.floor("h")
# airline / origin_airport are already category from Cell 1 - no re-cast needed

df["origin_hourly_sched_count"] = (
    tmp.groupby(["origin_airport", "sched_hour_bucket"], observed=True)["scheduled_departure"]
       .transform("count")
       .astype("float32")
       .values
)
gc.collect()

global_mean_delay = tmp["departure_delay_minutes"].mean()

stats_aa = (
    tmp.groupby(["airline", "origin_airport", "actual_hour_bucket"], observed=True)["departure_delay_minutes"]
       .mean()
)
lookup_index_aa = pd.MultiIndex.from_arrays([
    tmp["airline"], tmp["origin_airport"], tmp["sched_hour_bucket"] - pd.Timedelta(hours=1),
])
df["airline_airport_prior_hour_delay"] = (
    stats_aa.reindex(lookup_index_aa).fillna(global_mean_delay).astype("float32").values
)
del stats_aa, lookup_index_aa
gc.collect()

stats_a = (
    tmp.groupby(["origin_airport", "actual_hour_bucket"], observed=True)["departure_delay_minutes"]
       .mean()
)
lookup_index_a = pd.MultiIndex.from_arrays([
    tmp["origin_airport"], tmp["sched_hour_bucket"] - pd.Timedelta(hours=1),
])
df["airport_prior_hour_delay"] = (
    stats_a.reindex(lookup_index_a).fillna(global_mean_delay).astype("float32").values
)
del stats_a, lookup_index_a, tmp
gc.collect()

print(df[["origin_hourly_sched_count", "airline_airport_prior_hour_delay", "airport_prior_hour_delay"]].describe())
```

---

## Cell 7 — Incoming Aircraft Delay (late-aircraft-arrival proxy via merge_asof)

Category-aligned join keys instead of `str` casts — fixes the same `MergeError` risk at a
fraction of the memory.

```python
# ============================================================================
# CELL 7 — INCOMING AIRCRAFT DELAY (LATE-AIRCRAFT-ARRIVAL PROXY VIA MERGE_ASOF)
# ============================================================================

def align_categories(s1, s2):
    cats = sorted(set(s1.astype("category").cat.categories) | set(s2.astype("category").cat.categories))
    dtype = pd.CategoricalDtype(categories=cats)
    return s1.astype(dtype), s2.astype(dtype)

arrivals = df[["airline", "destination_airport", "actual_arrival", "scheduled_arrival"]].copy()
arrivals["arrival_delay_minutes"] = (
    (arrivals["actual_arrival"] - arrivals["scheduled_arrival"]).dt.total_seconds() / 60
).astype("float32")
arrivals = arrivals.rename(columns={"destination_airport": "airport"})
arrivals = arrivals.dropna(subset=["actual_arrival"])
arrivals = arrivals[["airline", "airport", "actual_arrival", "arrival_delay_minutes"]]
arrivals = arrivals.sort_values("actual_arrival")
gc.collect()

departures = df[["airline", "origin_airport", "scheduled_departure"]].copy()
departures = departures.rename(columns={"origin_airport": "airport"})
departures["orig_idx"] = df.index
departures = departures.sort_values("scheduled_departure")
gc.collect()

arrivals["airline"], departures["airline"] = align_categories(arrivals["airline"], departures["airline"])
arrivals["airport"], departures["airport"] = align_categories(arrivals["airport"], departures["airport"])

merged = pd.merge_asof(
    departures, arrivals,
    left_on="scheduled_departure", right_on="actual_arrival",
    by=["airline", "airport"], direction="backward",
    tolerance=pd.Timedelta(hours=6),
)
merged = merged.set_index("orig_idx").sort_index()

df["incoming_aircraft_delay"] = merged["arrival_delay_minutes"].astype("float32")
df["has_incoming_match"] = merged["arrival_delay_minutes"].notna().astype("float32")

global_mean_delay = df["departure_delay_minutes"].mean()
df["incoming_aircraft_delay"] = df["incoming_aircraft_delay"].fillna(global_mean_delay).astype("float32")

del arrivals, departures, merged
gc.collect()

print(df[["incoming_aircraft_delay", "has_incoming_match"]].describe())
print("match rate:", df["has_incoming_match"].mean())
```

---

## Cell 7b — Route Trailing-30-Day Delay (merge_asof, causal)

Aggregates delay by exact route, then attaches the trailing 30-day rolling average available
as of *the day before* each flight — never same-day data. Runs on a 3-column subset instead
of the full `df`.

```python
# ============================================================================
# CELL 7B — ROUTE TRAILING-30-DAY DELAY (MERGE_ASOF, CAUSAL)
# ============================================================================

route_sub = df[["route", "scheduled_departure", "departure_delay_minutes"]].copy()

route_daily = (
    route_sub.set_index("scheduled_departure")
      .groupby("route", observed=True)["departure_delay_minutes"]
      .resample("1D").mean()
      .reset_index()
      .rename(columns={"scheduled_departure": "date",
                        "departure_delay_minutes": "route_daily_mean_delay"})
)
del route_sub
gc.collect()
route_daily = route_daily.sort_values(["route", "date"])

route_daily["route_trailing30d_delay"] = (
    route_daily.groupby("route", observed=True)["route_daily_mean_delay"]
    .transform(lambda s: s.rolling(30, min_periods=3).mean())
)

# shift availability forward by 1 day so a flight can only see PRIOR days' data
route_daily["available_from"] = route_daily["date"] + pd.Timedelta(days=1)
route_daily = route_daily.sort_values("available_from")

flights_route = df[["route", "scheduled_departure"]].copy()
flights_route = flights_route.sort_values("scheduled_departure")

flights_route["route"], route_daily["route"] = align_categories(flights_route["route"], route_daily["route"])

merged_route = pd.merge_asof(
    flights_route, route_daily[["route", "available_from", "route_trailing30d_delay"]],
    left_on="scheduled_departure", right_on="available_from",
    by="route", direction="backward",
    tolerance=pd.Timedelta(days=35),
)
merged_route.index = flights_route.index

global_mean_delay = df["departure_delay_minutes"].mean()
df["route_trailing30d_delay"] = (
    merged_route["route_trailing30d_delay"].reindex(df.index)
    .fillna(global_mean_delay).astype("float32")
)

del route_daily, flights_route, merged_route
gc.collect()
print(df["route_trailing30d_delay"].describe())
```

---

## Cell 7c — Flight-Number Previous-Occurrence Delay (aircraft-rotation proxy)

No tail number available, so this is the closest causal proxy: how delayed was *this exact
flight number's* last occurrence. Runs on a 4-column subset instead of the full `df`.

```python
# ============================================================================
# CELL 7C — FLIGHT-NUMBER PREVIOUS-OCCURRENCE DELAY (AIRCRAFT-ROTATION PROXY)
# ============================================================================

fn_sub = df[["airline", "flight_number", "scheduled_departure", "departure_delay_minutes"]].copy()
fn_sub = fn_sub.sort_values("scheduled_departure")

prev_delay = fn_sub.groupby(["airline", "flight_number"], observed=True)["departure_delay_minutes"].shift(1)
prev_time  = fn_sub.groupby(["airline", "flight_number"], observed=True)["scheduled_departure"].shift(1)

global_mean_delay = df["departure_delay_minutes"].mean()

df["flight_number_prev_delay"] = (
    prev_delay.reindex(df.index).fillna(global_mean_delay).astype("float32")
)
# gap in days since that previous occurrence - a stale lag (e.g. seasonal route,
# 200 days ago) is much less informative than a fresh one
df["flight_number_prev_gap_days"] = (
    (df["scheduled_departure"] - prev_time.reindex(df.index)).dt.total_seconds() / 86400
).fillna(999).astype("float32")

del fn_sub, prev_delay, prev_time
gc.collect()
print(df[["flight_number_prev_delay", "flight_number_prev_gap_days"]].describe())
```

---

## Cell 7d — Extended Airport Delay Trend (3h rolling + slope)

Round 4 only looked at a single preceding hour (`airport_prior_hour_delay`). This adds a
longer rolling window and a trend signal (congestion building or clearing). Runs on a
3-column subset instead of the full `df`.

```python
# ============================================================================
# CELL 7D — EXTENDED AIRPORT DELAY TREND (3H ROLLING + SLOPE)
# ============================================================================

ap_sub = df[["origin_airport", "actual_departure", "departure_delay_minutes"]].copy()

airport_hourly = (
    ap_sub.set_index("actual_departure")
      .groupby("origin_airport", observed=True)["departure_delay_minutes"]
      .resample("1h").mean()
      .reset_index()
      .rename(columns={"actual_departure": "hour_bucket",
                        "departure_delay_minutes": "hourly_mean_delay"})
)
del ap_sub
gc.collect()
airport_hourly = airport_hourly.sort_values(["origin_airport", "hour_bucket"])

airport_hourly["airport_trailing3h_delay"] = (
    airport_hourly.groupby("origin_airport", observed=True)["hourly_mean_delay"]
    .transform(lambda s: s.rolling(3, min_periods=1).mean())
)
# trend: current-hour-ish level minus 3h level (positive = worsening)
airport_hourly["airport_delay_trend"] = (
    airport_hourly["hourly_mean_delay"] - airport_hourly["airport_trailing3h_delay"]
)

# availability = 1 hour after the bucket (matches the existing causal convention)
airport_hourly["available_from"] = airport_hourly["hour_bucket"] + pd.Timedelta(hours=1)
airport_hourly = airport_hourly.sort_values("available_from")

flights_ap = df[["origin_airport", "scheduled_departure"]].copy()
flights_ap = flights_ap.sort_values("scheduled_departure")

flights_ap["origin_airport"], airport_hourly["origin_airport"] = align_categories(
    flights_ap["origin_airport"], airport_hourly["origin_airport"]
)

merged_ap = pd.merge_asof(
    flights_ap, airport_hourly[["origin_airport", "available_from",
                                  "airport_trailing3h_delay", "airport_delay_trend"]],
    left_on="scheduled_departure", right_on="available_from",
    by="origin_airport", direction="backward",
    tolerance=pd.Timedelta(hours=6),
)
merged_ap.index = flights_ap.index

global_mean_delay = df["departure_delay_minutes"].mean()
df["airport_trailing3h_delay"] = (
    merged_ap["airport_trailing3h_delay"].reindex(df.index).fillna(global_mean_delay).astype("float32")
)
df["airport_delay_trend"] = (
    merged_ap["airport_delay_trend"].reindex(df.index).fillna(0).astype("float32")
)

del airport_hourly, flights_ap, merged_ap
gc.collect()
print(df[["airport_trailing3h_delay", "airport_delay_trend"]].describe())
```

---

## Cell 7e — Holiday Flag (via `country` + date)

Requires `pip install holidays`. Assumes `country` maps to a valid ISO-3166 alpha-2 code
recognized by the `holidays` package — check `df["country"].unique()` first and adjust the
mapping if needed. Runs on a 2-column working frame instead of the full `df`.

```python
# ============================================================================
# CELL 7E — HOLIDAY FLAG (VIA COUNTRY + DATE)
# ============================================================================

import holidays

country_map = {
    # fill in / adjust based on df["country"].unique()
    "United States": "US", "USA": "US", "US": "US",
}
def to_iso(c):
    return country_map.get(c, c)

work = df[["country", "scheduled_departure"]].copy()
work["_country_iso"] = work["country"].astype(str).map(to_iso)
work["_flight_date"] = work["scheduled_departure"].dt.date
work["_year"] = work["scheduled_departure"].dt.year

is_holiday = pd.Series(False, index=df.index)
for (country_iso, year), group in work.groupby(["_country_iso", "_year"], observed=True):
    try:
        cal = holidays.country_holidays(country_iso, years=int(year))
    except Exception:
        continue
    is_holiday.loc[group.index] = group["_flight_date"].isin(cal).values

df["is_holiday"] = is_holiday.astype("float32")
del work, is_holiday
gc.collect()

print("holiday rate:", df["is_holiday"].mean())
```

---

## Cell 8 — Reattach All Engineered + Weather Features to Splits

`df` is not needed again after this — freed here instead of carried through training.

```python
# ============================================================================
# CELL 8 — REATTACH ALL ENGINEERED + WEATHER FEATURES TO SPLITS
# ============================================================================

NEW_FEATURES = [
    "origin_hourly_sched_count", "airline_airport_prior_hour_delay", "airport_prior_hour_delay",
    "incoming_aircraft_delay", "has_incoming_match",
    # Round 5 additions
    "route_trailing30d_delay",
    "flight_number_prev_delay", "flight_number_prev_gap_days",
    "airport_trailing3h_delay", "airport_delay_trend",
    "is_holiday",
]

for col in NEW_FEATURES + WEATHER_FEATURES:
    train_df[col] = df.loc[train_df.index, col].astype("float32")
    val_df[col] = df.loc[val_df.index, col].astype("float32")
    test_df[col] = df.loc[test_df.index, col].astype("float32")

# df is not touched again anywhere downstream - free it now instead of
# carrying a full extra copy of the dataset through training
del df
gc.collect()

print("train:", train_df.shape, " val:", val_df.shape, " test:", test_df.shape)
print("NaN check:", train_df[NEW_FEATURES + WEATHER_FEATURES].isna().sum().sum())
```

**Do not add `flight_status` or `cancellation_reason_available` to `FINAL_FEATURES`** — both
are almost certainly only resolved at or after actual departure, so they would leak the
target. (They're not even loaded in Cell 1 in this version.)

---

## Cell 9 — Final Feature Matrices

Wide split frames are dropped once `X`/`y` are extracted.

```python
# ============================================================================
# CELL 9 — FINAL FEATURE MATRICES
# ============================================================================

FINAL_FEATURES = [
    "airline_te", "origin_airport_te", "destination_airport_te",
    "flight_number_freq",
    "scheduled_flight_duration_minutes",
    "hour_of_day_sin", "hour_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_redeye",
] + WEATHER_FEATURES + NEW_FEATURES

X_train = train_df[FINAL_FEATURES].astype("float32")
y_train = train_df[TARGET].astype("float32")
X_val = val_df[FINAL_FEATURES].astype("float32")
y_val = val_df[TARGET].astype("float32")
X_test = test_df[FINAL_FEATURES].astype("float32")
y_test = test_df[TARGET].astype("float32")

print("train:", X_train.shape, " val:", X_val.shape, " test:", X_test.shape)
print("NaN check:", X_train.isna().sum().sum(), X_val.isna().sum().sum(), X_test.isna().sum().sum())

# free the wide DataFrames now that X/y are extracted - only run this if you
# don't need train_df/val_df/test_df's other columns again later
del train_df, val_df, test_df, trainval_df
gc.collect()
```

---

## Cell 10 — Train All Algorithms

```python
# ============================================================================
# CELL 10 — TRAIN ALL ALGORITHMS
# ============================================================================

import time
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

results = {}

def evaluate_model(name, y_true, y_pred, train_time=None):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "train_time_sec": train_time}
    print(f"{name} -> RMSE: {rmse:.3f}  MAE: {mae:.3f}  R2: {r2:.4f}"
          + (f"  ({train_time:.1f}s)" if train_time else ""))
    return results[name]

# Ridge
t0 = time.time()
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
evaluate_model("Ridge", y_val, ridge.predict(X_val), time.time() - t0)

# Linear Regression
t0 = time.time()
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate_model("Linear Regression", y_val, lr.predict(X_val), time.time() - t0)

# XGBoost (GPU)
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", device="cuda",
    random_state=42, n_jobs=-1,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
evaluate_model("XGBoost (GPU)", y_val, xgb_model.predict(X_val), time.time() - t0)

# LightGBM (GPU)
t0 = time.time()
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    device="gpu", random_state=42, n_jobs=-1,
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.log_evaluation(100)])
evaluate_model("LightGBM (GPU)", y_val, lgb_model.predict(X_val), time.time() - t0)

# CatBoost (GPU)
t0 = time.time()
cat_model = CatBoostRegressor(
    iterations=500, depth=8, learning_rate=0.05,
    task_type="GPU", devices="0:1",
    random_state=42, verbose=100,
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
evaluate_model("CatBoost (GPU)", y_val, cat_model.predict(X_val), time.time() - t0)

print("\n" + "=" * 70)
print(pd.DataFrame(results).T.sort_values("RMSE"))
```

---

## Cell 11 — Feature Importance Check

```python
# ============================================================================
# CELL 11 — FEATURE IMPORTANCE CHECK
# ============================================================================

importance = pd.Series(cat_model.feature_importances_, index=FINAL_FEATURES).sort_values(ascending=False)
print(importance)
```

---

## Round 5 Results (Validation Set)

| Model | RMSE | MAE | R² | Train Time |
|---|---|---|---|---|
| **XGBoost (GPU)** | **51.03** | **20.38** | **0.17** | 120.5s |
| LightGBM (GPU) | 51.09 | 20.55 | 0.16 | 335.4s |
| CatBoost (GPU) | 51.21 | 20.57 | 0.16 | 73.7s |
| Linear Regression | 54.02 | 22.91 | 0.06 | 13.4s |
| Ridge | 54.02 | 22.91 | 0.06 | 4.2s |

**Round 4 → Round 5 delta (XGBoost, best model each round):**

| Round | RMSE | MAE | R² |
|---|---|---|---|
| Round 4 | 52.68 | 21.61 | 0.11 |
| Round 5 | 51.03 | 20.38 | 0.17 |
| **Δ** | **-1.65** | **-1.23** | **+0.06** |

The historical/lag features (route trailing delay, flight-number rotation proxy, extended
airport trend, holiday flag) delivered the single largest jump in the project so far —
R² moved from 0.11 to 0.17, a ~55% relative improvement, confirming the hypothesis that
route/schedule-specific history carries signal that airline/airport-level target encoding
alone can't capture. Ridge/Linear also improved (0.04 → 0.06) since some of the new features
are more linearly related to delay than the raw categoricals were.

Two runtime notes from the log, not accuracy-affecting:
- The `LinAlgWarning: Ill-conditioned matrix` from Ridge/Linear Regression suggests some
  near-collinearity in `FINAL_FEATURES` (likely between `airport_prior_hour_delay` /
  `airport_trailing3h_delay` / `airport_delay_trend`, or between the two route/flight lag
  features). Doesn't affect the tree models; if you want cleaner linear coefficients later,
  consider dropping one of the correlated pair or adding a small `alpha` bump to Ridge.
- The XGBoost `device mismatch` warning (`cuda:0` vs CPU input) means prediction silently
  fell back to `DMatrix`/CPU — training was still on GPU, so the 120.5s time and RMSE are
  unaffected, but converting `X_val`/`X_test` to a `QuantileDMatrix` or a cupy/cudf array
  before `.predict()` would remove the warning and likely speed up inference if you're
  running that repeatedly.

---

## Next Steps to Consider for Round 6

- Investigate the Ridge/Linear collinearity flagged above (VIF check on the airport-trend
  trio and the two route/flight-number lag features).
- Fix the XGBoost GPU predict-device warning for faster inference.
- Try the log-transform / Tweedie objective noted as optional in Round 5 — not yet tested
  against these results.
- Check Cell 11 feature importances to confirm which of the five new Round 5 features are
  actually pulling weight vs. just adding training time (LightGBM in particular got slower
  without matching XGBoost's improvement).
