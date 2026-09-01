# Flight Departure Delay Prediction — Round 4 Pipeline

Full pipeline: time-based split → airport code normalization → target/frequency/cyclical encoding → weather features → engineered operational features (congestion, cascading delay proxy, incoming aircraft delay) → model training (Ridge, Linear Regression, XGBoost, LightGBM, CatBoost).

**Current best result (Round 4): XGBoost — RMSE 52.68, MAE 21.61, R² 0.11**

Progression:
- Round 1 (schedule only): R² ≈ 0.04
- Round 2 (+ weather): R² ≈ 0.04
- Round 3 (+ hourly congestion/cascading avg): R² ≈ 0.08
- Round 4 (+ incoming aircraft delay via merge_asof): R² ≈ 0.11

---

## Cell 1 — Load Data & Time-Based Split

```python
import pandas as pd
import numpy as np
import gc

# df = pd.read_parquet("your_file.parquet")  # load your full dataset

FEATURES_CAT = ["airline", "flight_number", "origin_airport", "destination_airport"]
FEATURES_NUM = [
    "scheduled_flight_duration_minutes",
    "hour_of_day", "day_of_week", "month",
    "is_weekend", "is_redeye",
]
WEATHER_FEATURES = ["temperature_2m_c", "windspeed_10m_ms", "visibility_m", "cloudcover_low_pct"]
TARGET = "departure_delay_minutes"
TIME_COL = "scheduled_departure"

df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"])
df["actual_arrival"] = pd.to_datetime(df["actual_arrival"])
df["actual_departure"] = pd.to_datetime(df["actual_departure"])

train_df = df[(df[TIME_COL] >= "2022-01-01") & (df[TIME_COL] <= "2024-12-31")].copy()
val_df   = df[(df[TIME_COL] >= "2025-01-01") & (df[TIME_COL] <= "2025-12-31")].copy()
test_df  = df[(df[TIME_COL] >= "2026-01-01") & (df[TIME_COL] <  "2026-06-02")].copy()

trainval_df = pd.concat([train_df, val_df], axis=0)

print("train:", train_df.shape, " val:", val_df.shape, " test:", test_df.shape)
```

---

## Cell 2 — Normalize Airport Codes (ICAO → IATA)

```python
airports = pd.read_csv("/kaggle/input/datasets/tjaycuz/airports/airports.csv")
icao_to_iata = (
    airports[["icao_code", "iata_code"]]
    .dropna()
    .set_index("icao_code")["iata_code"]
    .to_dict()
)

def normalize_airport(code):
    code = str(code).strip().upper()
    if len(code) == 3:
        return code
    if len(code) == 4:
        return icao_to_iata.get(code, code)
    return code

for d in (train_df, val_df, test_df):
    for col in ["origin_airport", "destination_airport"]:
        d[col] = d[col].map(normalize_airport).astype("category")

# also normalize on the full df — needed for feature engineering steps later
for col in ["origin_airport", "destination_airport"]:
    df[col] = df[col].map(normalize_airport).astype("category")

trainval_df = pd.concat([train_df, val_df], axis=0)

for col in ["origin_airport", "destination_airport"]:
    print(f"\n{col}")
    print(train_df[col].astype(str).str.len().value_counts().sort_index())
```

---

## Cell 3 — Target Encoding (airline, origin_airport, destination_airport)

Leakage-safe: (a) K-fold out-of-fold within train, (b) val encoded from train-only stats, (c) test encoded from train+val stats.

```python
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
freq_map_train = train_df["flight_number"].value_counts(normalize=True)
freq_map_trainval = trainval_df["flight_number"].value_counts(normalize=True)

train_df["flight_number_freq"] = train_df["flight_number"].map(freq_map_train).astype("float32")
val_df["flight_number_freq"] = val_df["flight_number"].map(freq_map_train).fillna(0).astype("float32")
test_df["flight_number_freq"] = test_df["flight_number"].map(freq_map_trainval).fillna(0).astype("float32")
```

---

## Cell 5 — Cyclical Encoding (hour, day_of_week, month)

```python
def add_cyclical(d, col, period):
    d[f"{col}_sin"] = np.sin(2 * np.pi * d[col] / period).astype("float32")
    d[f"{col}_cos"] = np.cos(2 * np.pi * d[col] / period).astype("float32")

for d in (train_df, val_df, test_df):
    add_cyclical(d, "hour_of_day", 24)
    add_cyclical(d, "day_of_week", 7)
    add_cyclical(d, "month", 12)
```

---

## Cell 6 — Feature Engineering: Airport Congestion + Cascading Delay Proxy (memory-efficient)

```python
cols_needed = ["airline", "origin_airport", "scheduled_departure",
               "actual_departure", "departure_delay_minutes"]
tmp = df[cols_needed].copy()

tmp["sched_hour_bucket"] = tmp["scheduled_departure"].dt.floor("h")
tmp["actual_hour_bucket"] = tmp["actual_departure"].dt.floor("h")
tmp["airline"] = tmp["airline"].astype("category")
tmp["origin_airport"] = tmp["origin_airport"].astype("category")

# 6a. Airport congestion — count of flights scheduled from same airport, same hour
df["origin_hourly_sched_count"] = (
    tmp.groupby(["origin_airport", "sched_hour_bucket"], observed=True)["scheduled_departure"]
       .transform("count")
       .astype("float32")
       .values
)
gc.collect()

# 6b. Cascading delay proxy — avg delay in the PRECEDING hour, causal by construction
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

Cast categorical join keys to `str` before `merge_asof` — mismatched `CategoricalDtype` categories between origin/destination columns cause a `MergeError` otherwise.

```python
arrivals = df[["airline", "destination_airport", "actual_arrival", "scheduled_arrival"]].copy()
arrivals["arrival_delay_minutes"] = (
    (arrivals["actual_arrival"] - arrivals["scheduled_arrival"]).dt.total_seconds() / 60
).astype("float32")
arrivals = arrivals.rename(columns={"destination_airport": "airport"})
arrivals = arrivals.dropna(subset=["actual_arrival"])
arrivals = arrivals[["airline", "airport", "actual_arrival", "arrival_delay_minutes"]]

arrivals["airline"] = arrivals["airline"].astype(str)
arrivals["airport"] = arrivals["airport"].astype(str)
arrivals = arrivals.sort_values("actual_arrival")
gc.collect()

departures = df[["airline", "origin_airport", "scheduled_departure"]].copy()
departures = departures.rename(columns={"origin_airport": "airport"})
departures["airline"] = departures["airline"].astype(str)
departures["airport"] = departures["airport"].astype(str)
departures["orig_idx"] = df.index
departures = departures.sort_values("scheduled_departure")
gc.collect()

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

## Cell 8 — Reattach All Engineered + Weather Features to Splits

```python
NEW_FEATURES = [
    "origin_hourly_sched_count", "airline_airport_prior_hour_delay", "airport_prior_hour_delay",
    "incoming_aircraft_delay", "has_incoming_match",
]

for col in NEW_FEATURES + WEATHER_FEATURES:
    train_df[col] = df.loc[train_df.index, col].astype("float32")
    val_df[col] = df.loc[val_df.index, col].astype("float32")
    test_df[col] = df.loc[test_df.index, col].astype("float32")

gc.collect()
print("train:", train_df.shape, " val:", val_df.shape, " test:", test_df.shape)
print("NaN check:", train_df[NEW_FEATURES + WEATHER_FEATURES].isna().sum().sum())
```

---

## Cell 9 — Final Feature Matrices

```python
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
```

---

## Cell 10 — Train All Algorithms

```python
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
importance = pd.Series(cat_model.feature_importances_, index=FINAL_FEATURES).sort_values(ascending=False)
print(importance)
```

---

## Round 4 Results (Validation Set)

| Model | RMSE | MAE | R² | Train Time |
|---|---|---|---|---|
| XGBoost (GPU) | 52.68 | 21.61 | 0.11 | 99.8s |
| CatBoost (GPU) | 52.80 | 21.80 | 0.11 | 69.9s |
| LightGBM (GPU) | 52.85 | 21.86 | 0.10 | 319.7s |
| Linear Regression | 54.85 | 23.45 | 0.04 | 9.9s |
| Ridge | 54.85 | 23.45 | 0.04 | 3.3s |
