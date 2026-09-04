# Round 7 Pipeline — Flight Departure Delay Prediction (2022-2025, with TAIL_NUM)

## What Changed From Round 6

1. **Source data:** switched to the fully-cleaned 2022-2025 dataset with `TAIL_NUM`
   merged in (`df-2022-2025-handled-missing-set-bounds-with-tail`), after fixing a bug
   where an earlier cell was accidentally pointed at a pre-cleaning file (caught via a
   NaT count that exactly matched the raw missingness table).
2. **Split:** changed from the old 3yr/1yr/6mo scheme to full calendar years —
   Train 2022-2023 / Val 2024 / Test 2025 — to avoid seasonal bias in val/test.
3. **TAIL_NUM added** as a categorical feature, target-encoded the same way as
   `airline`/`origin_airport`/`destination_airport` (5-fold out-of-fold, smoothing=10).
4. **New feature group: aircraft rotation / cascading risk**, built from `TAIL_NUM` +
   `scheduled_departure` (Cell 8) — `aircraft_leg_number_of_day`,
   `scheduled_gap_since_prev_leg`, `is_first_flight_of_day_for_tail`, and a
   `has_rotation_data` flag for the `"Unknown"` tail bucket.
5. **Memory-crash fix (Cells 14-15):** the notebook hit an out-of-memory restart after
   Cell 14 when all five models were trained back-to-back in one cell on top of an
   already-large in-memory pipeline. Root causes and fixes:
   - `xgb.QuantileDMatrix` was being built and never used — removed.
   - sklearn's `Ridge`/`LinearRegression` defaults copy `X` and upcast to `float64`,
     silently doubling+ memory on a 15M-row matrix — replaced with a hand-rolled,
     chunked closed-form solve (`XtX`, `Xty` accumulated in 1M-row chunks via a
     memory-mapped `.npz`, never loading the full training matrix as `float64` at once).
   - **Matrices persisted to disk** (`/kaggle/working/matrices.npz` +
     `final_features.json`) at the end of Cell 14, so the kernel can be **restarted**
     before Cell 15, and each model now runs in its own fresh-kernel cell — no
     dependency on Cells 1-14 still being in memory, no accumulation of five models'
     footprints in one session.
   - Each model cell reloads only what it needs from disk, trains, evaluates, and
     writes its result to `/kaggle/working/results.json` (append-safe across restarts),
     then can be discarded.
   - Feature importance (CatBoost) also written to disk
     (`/kaggle/working/feature_importance.json`) so Cell 16 has no dependency on the
     model object still being alive.

---

## Pipeline Cells (actual code)

**Cell 0 — Imports & display options**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
pd.set_option("display.float_format", "{:,.2f}".format)
```

---

**Cell 1 — Load Data & Time-Based Split**

```python
import gc

FEATURES_CAT = ["airline", "flight_number", "origin_airport", "destination_airport", "TAIL_NUM"]
FEATURES_NUM = [
    "scheduled_flight_duration_minutes",
    "hour_of_day", "day_of_week", "month",
    "is_weekend", "is_redeye",
]
WEATHER_FEATURES = ["temperature_2m_c", "windspeed_10m_ms", "visibility_m", "cloudcover_low_pct"]
TARGET = "departure_delay_minutes"
TIME_COL = "scheduled_departure"

NEEDED_COLS = list(set(
    FEATURES_CAT + FEATURES_NUM + WEATHER_FEATURES +
    [TARGET, TIME_COL, "scheduled_arrival", "actual_arrival", "actual_departure",
     "route", "country"]
))

df = pd.read_parquet(
    "/kaggle/input/datasets/tjaycuz/df-2022-2025-handled-missing-set-bounds-with-tail",
    columns=NEEDED_COLS,
)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"])
df["actual_arrival"] = pd.to_datetime(df["actual_arrival"])
df["actual_departure"] = pd.to_datetime(df["actual_departure"])

for c in df.select_dtypes(include="float64").columns:
    df[c] = df[c].astype("float32")
for c in df.select_dtypes(include="int64").columns:
    df[c] = df[c].astype("int32")

for c in ["airline", "flight_number", "origin_airport", "destination_airport", "route", "country", "TAIL_NUM"]:
    if c in df.columns:
        df[c] = df[c].astype("category")

train_mask = (df[TIME_COL] >= "2022-01-01") & (df[TIME_COL] <= "2023-12-31")
val_mask   = (df[TIME_COL] >= "2024-01-01") & (df[TIME_COL] <= "2024-12-31")
test_mask  = (df[TIME_COL] >= "2025-01-01") & (df[TIME_COL] <= "2025-12-31")

train_df = df.loc[train_mask].copy()
val_df   = df.loc[val_mask].copy()
test_df  = df.loc[test_mask].copy()

del train_mask, val_mask, test_mask
gc.collect()

# Built ONCE here. From here on, downstream cells update individual columns
# in place (single-column concat) instead of rebuilding the whole frame.
trainval_df = pd.concat([train_df, val_df], axis=0)
print("train:", train_df.shape, " val:", val_df.shape, " test:", test_df.shape)
```

---

**Cell 2 — Normalize Airport Codes (ICAO -> IATA)**

```python
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

for col in ["origin_airport", "destination_airport"]:
    df[col] = df[col].map(normalize_airport).astype("category")

for d in (train_df, val_df, test_df):
    for col in ["origin_airport", "destination_airport"]:
        d[col] = df.loc[d.index, col].astype("category")

# Update only the two changed columns in trainval_df instead of re-concatenating
# the entire frame (which was allocating a full 20M-row copy each time).
for col in ["origin_airport", "destination_airport"]:
    trainval_df[col] = pd.concat([train_df[col], val_df[col]])

for col in ["origin_airport", "destination_airport"]:
    print(f"\n{col}")
    print(train_df[col].astype(str).str.len().value_counts().sort_index())
```

---

**Cell 3 — Target Encoding (airline, origin_airport, destination_airport, month_airport, TAIL_NUM)**

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
        del fold_train, stats, smooth_mean, val_slice
    return train_enc.astype("float32")

def target_encode_from_reference(reference, apply_to, col, target, smoothing=10):
    global_mean = reference[target].mean()
    stats = reference.groupby(col, observed=True)[target].agg(["mean", "count"])
    smooth_mean = (stats["mean"] * stats["count"] + global_mean * smoothing) / (stats["count"] + smoothing)
    return apply_to[col].map(smooth_mean).fillna(global_mean).astype("float32")

# trainval_df here is the raw-column reference for test-set encoding (uses raw
# col + TARGET, not the *_te columns), so nothing needs adding to it yet.
for col in ["airline", "origin_airport", "destination_airport", "TAIL_NUM"]:
    train_df[f"{col}_te"] = kfold_target_encode_train(train_df, col, TARGET, n_splits=5, smoothing=10)
    val_df[f"{col}_te"] = target_encode_from_reference(train_df, val_df, col, TARGET, smoothing=10)
    test_df[f"{col}_te"] = target_encode_from_reference(trainval_df, test_df, col, TARGET, smoothing=10)
    gc.collect()

# month x airport seasonal interaction
for d in (train_df, val_df, test_df):
    d["month_airport"] = (d["month"].astype(str) + "_" + d["origin_airport"].astype(str)).astype("category")

train_df["month_airport_te"] = kfold_target_encode_train(train_df, "month_airport", TARGET, n_splits=5, smoothing=10)
val_df["month_airport_te"] = target_encode_from_reference(train_df, val_df, "month_airport", TARGET, smoothing=10)

# Add the new "month_airport" column to trainval_df as a single-column concat
# instead of rebuilding the whole frame from scratch.
trainval_df["month_airport"] = pd.concat([train_df["month_airport"], val_df["month_airport"]])
test_df["month_airport_te"] = target_encode_from_reference(trainval_df, test_df, "month_airport", TARGET, smoothing=10)
gc.collect()
```

---

**Cell 4 — Frequency Encoding (flight_number)**

```python
freq_map_train = train_df["flight_number"].value_counts(normalize=True)
freq_map_trainval = trainval_df["flight_number"].value_counts(normalize=True)

train_df["flight_number_freq"] = train_df["flight_number"].map(freq_map_train).astype("float32")
val_df["flight_number_freq"] = val_df["flight_number"].map(freq_map_train).fillna(0).astype("float32")
test_df["flight_number_freq"] = test_df["flight_number"].map(freq_map_trainval).fillna(0).astype("float32")

del freq_map_train, freq_map_trainval
gc.collect()

del trainval_df
gc.collect()
```

---

**Cell 5 — Cyclical Encoding**

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

**Cell 6 — Feature Engineering Helpers**

```python
DELAY_THRESHOLD_MIN = 15
GLOBAL_MEAN_DELAY = train_df[TARGET].mean()
GLOBAL_DELAY_RATE = (train_df[TARGET] >= DELAY_THRESHOLD_MIN).mean()

def align_categories(s1, s2):
    cats = sorted(set(s1.astype("category").cat.categories) | set(s2.astype("category").cat.categories))
    dtype = pd.CategoricalDtype(categories=cats)
    return s1.astype(dtype), s2.astype(dtype)

def build_bucketed_table(source_df, group_cols, time_col, value_col, freq):
    sub = source_df[group_cols + [time_col, value_col]].copy()
    sub["_is_delayed"] = (sub[value_col] >= DELAY_THRESHOLD_MIN).astype("float32")
    sub["_bucket"] = sub[time_col].dt.floor(freq)
    table = (
        sub.groupby(group_cols + ["_bucket"], observed=True)
           .agg(_mean=(value_col, "mean"), _rate=("_is_delayed", "mean"))
           .reset_index()
           .rename(columns={"_bucket": time_col})
    )
    del sub
    gc.collect()
    return table

def add_rolling(table, group_cols, src_col, windows, out_prefix, min_periods=None):
    g = table.groupby(group_cols, observed=True)[src_col]
    n_levels = len(group_cols)
    for w in windows:
        mp = min_periods if min_periods is not None else max(1, w // 3)
        table[f"{out_prefix}{w}"] = (
            g.rolling(w, min_periods=mp).mean()
             .reset_index(level=list(range(n_levels)), drop=True)
             .astype("float32")
        )
    return table

def causal_attach(df_, table, by_cols, table_time_col, base_time_col, feature_cols, fillna_map, tolerance):
    left = df_[by_cols + [base_time_col]].copy()
    left["_orig_idx"] = df_.index
    left = left.sort_values(base_time_col)
    right = table.sort_values(table_time_col)

    for c in by_cols:
        left[c], right[c] = align_categories(left[c], right[c])

    merged = pd.merge_asof(
        left, right[by_cols + [table_time_col] + feature_cols],
        left_on=base_time_col, right_on=table_time_col,
        by=by_cols, direction="backward", tolerance=tolerance,
    )
    merged = merged.set_index("_orig_idx").sort_index()

    out = pd.DataFrame(index=df_.index)
    for c in feature_cols:
        out[c] = merged[c].reindex(df_.index).fillna(fillna_map.get(c, 0)).astype("float32")
    del left, right, merged
    return out

def trailing_sched_count(source_df, group_cols, time_col, hours):
    sub = source_df[group_cols + [time_col]].copy()
    sub["_orig_idx"] = source_df.index
    sub = sub.sort_values(group_cols + [time_col])
    window = np.timedelta64(hours, "h")
    parts = []
    for _, g in sub.groupby(group_cols, observed=True):
        t = g[time_col].values.astype("datetime64[ns]")
        start = np.searchsorted(t, t - window, side="left")
        counts = (np.arange(len(t)) - start).astype("float32")
        parts.append(pd.Series(counts, index=g["_orig_idx"]))
    result = pd.concat(parts).reindex(source_df.index).astype("float32")
    del sub, parts
    return result
```

**Diagnostic (run before Cell 7)** — caught the wrong-source-file bug earlier this round:

```python
print("scheduled_departure NaT:", df["scheduled_departure"].isna().sum())
print("actual_departure NaT:", df["actual_departure"].isna().sum())
print("airline NaN:", df["airline"].isna().sum())
print("origin_airport NaN:", df["origin_airport"].isna().sum())
print("destination_airport NaN:", df["destination_airport"].isna().sum())
```

---

**Cell 7 — Airport Congestion + Incoming Aircraft Delay**

```python
gc.collect()

df["origin_hourly_sched_count"] = trailing_sched_count(df, ["origin_airport"], "scheduled_departure", 1)

hourly_aa = build_bucketed_table(df, ["airline", "origin_airport"], "actual_departure", TARGET, "1h")
hourly_aa["available_from"] = hourly_aa["actual_departure"] + pd.Timedelta(hours=1)
attach = causal_attach(
    df, hourly_aa, ["airline", "origin_airport"], "available_from", "scheduled_departure",
    ["_mean"], {"_mean": GLOBAL_MEAN_DELAY}, pd.Timedelta(hours=6),
)
df["airline_airport_prior_hour_delay"] = attach["_mean"]
del hourly_aa, attach
gc.collect()

hourly_a = build_bucketed_table(df, ["origin_airport"], "actual_departure", TARGET, "1h")
hourly_a["available_from"] = hourly_a["actual_departure"] + pd.Timedelta(hours=1)
attach = causal_attach(
    df, hourly_a, ["origin_airport"], "available_from", "scheduled_departure",
    ["_mean"], {"_mean": GLOBAL_MEAN_DELAY}, pd.Timedelta(hours=6),
)
df["airport_prior_hour_delay"] = attach["_mean"]
del attach
gc.collect()

arrivals = df[["airline", "destination_airport", "actual_arrival", "scheduled_arrival"]].copy()
arrivals["arrival_delay_minutes"] = ((arrivals["actual_arrival"] - arrivals["scheduled_arrival"]).dt.total_seconds() / 60).astype("float32")
arrivals = arrivals.rename(columns={"destination_airport": "airport"}).dropna(subset=["actual_arrival"])
arrivals = arrivals[["airline", "airport", "actual_arrival", "arrival_delay_minutes"]]

attach = causal_attach(
    df.rename(columns={"origin_airport": "airport"}), arrivals,
    ["airline", "airport"], "actual_arrival", "scheduled_departure",
    ["arrival_delay_minutes"], {"arrival_delay_minutes": GLOBAL_MEAN_DELAY}, pd.Timedelta(hours=6),
)
df["incoming_aircraft_delay"] = attach["arrival_delay_minutes"]
df["has_incoming_match"] = (df["incoming_aircraft_delay"] != GLOBAL_MEAN_DELAY).astype("float32")

del arrivals, attach, hourly_a
gc.collect()
print(df[["origin_hourly_sched_count", "airline_airport_prior_hour_delay",
          "airport_prior_hour_delay", "incoming_aircraft_delay"]].describe())
```

---

**Cell 8 — NEW: Aircraft Rotation / Cascading Risk Features (TAIL_NUM)**

```python
df = df.sort_values(["TAIL_NUM", "scheduled_departure"])

_flight_date = df["scheduled_departure"].dt.floor("D")
df["_tail_day_key"] = (
    df["TAIL_NUM"].astype(str) + "_" + _flight_date.astype(str)
).astype("category")

grp = df.groupby("_tail_day_key", observed=True)

df["aircraft_leg_number_of_day"] = (grp.cumcount() + 1).astype("float32")

prev_sched_dep = grp["scheduled_departure"].shift(1)
df["scheduled_gap_since_prev_leg"] = (
    (df["scheduled_departure"] - prev_sched_dep).dt.total_seconds() / 60
).astype("float32")

df["is_first_flight_of_day_for_tail"] = (df["aircraft_leg_number_of_day"] == 1).astype("float32")

# has_rotation_data: False for the "Unknown" TAIL_NUM bucket -- no real aircraft to
# build a rotation sequence for. Neutral fallback values below, flag carries the info.
has_rotation = (df["TAIL_NUM"].astype(str) != "Unknown")
df["has_rotation_data"] = has_rotation.astype("float32")

df.loc[~has_rotation, "aircraft_leg_number_of_day"] = 1.0
df.loc[~has_rotation, "scheduled_gap_since_prev_leg"] = -1.0
df.loc[~has_rotation, "is_first_flight_of_day_for_tail"] = 1.0

# real first-leg-of-day rows also get -1 sentinel (no prior leg today, not NaN)
df["scheduled_gap_since_prev_leg"] = df["scheduled_gap_since_prev_leg"].fillna(-1.0).astype("float32")

df = df.drop(columns=["_tail_day_key"])
del _flight_date, grp, prev_sched_dep, has_rotation
gc.collect()

print(df[["aircraft_leg_number_of_day", "scheduled_gap_since_prev_leg",
          "is_first_flight_of_day_for_tail", "has_rotation_data"]].describe())
print("\nRotation data coverage:", df["has_rotation_data"].mean())
```

---

**Cell 9 — Route & Flight-Number History**

```python
route_daily = build_bucketed_table(df, ["route"], "scheduled_departure", TARGET, "1D")
route_daily = add_rolling(route_daily, ["route"], "_mean", [30], "route_trailing", min_periods=3)
route_daily = add_rolling(route_daily, ["route"], "_rate", [1], "route_delay_rate_")
route_daily["available_from"] = route_daily["scheduled_departure"] + pd.Timedelta(days=1)

attach = causal_attach(
    df, route_daily, ["route"], "available_from", "scheduled_departure",
    ["route_trailing30", "route_delay_rate_1"],
    {"route_trailing30": GLOBAL_MEAN_DELAY, "route_delay_rate_1": GLOBAL_DELAY_RATE},
    pd.Timedelta(days=35),
)
df["route_trailing30d_delay"] = attach["route_trailing30"]
df["route_delay_rate_24h"] = attach["route_delay_rate_1"]
del route_daily, attach
gc.collect()

df["route_flights_trailing3h"] = trailing_sched_count(df, ["route"], "scheduled_departure", 3)
df["route_flights_trailing24h"] = trailing_sched_count(df, ["route"], "scheduled_departure", 24)
gc.collect()

fn_sub = df[["airline", "flight_number", "scheduled_departure", TARGET]].copy()
fn_sub = fn_sub.sort_values("scheduled_departure")
grp = fn_sub.groupby(["airline", "flight_number"], observed=True)[TARGET]

prev_delay = grp.shift(1)
prev_time = fn_sub.groupby(["airline", "flight_number"], observed=True)["scheduled_departure"].shift(1)
df["flight_number_prev_delay"] = prev_delay.reindex(df.index).fillna(GLOBAL_MEAN_DELAY).astype("float32")
df["flight_number_prev_gap_days"] = (
    (df["scheduled_departure"] - prev_time.reindex(df.index)).dt.total_seconds() / 86400
).fillna(999).astype("float32")

for w, name in [(3, "flight_number_last3_mean_delay"),
                (5, "flight_number_last5_mean_delay"),
                (10, "flight_number_last10_mean_delay")]:
    roll = prev_delay.groupby([fn_sub["airline"], fn_sub["flight_number"]], observed=True) \
                      .rolling(w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    df[name] = roll.reindex(df.index).fillna(GLOBAL_MEAN_DELAY).astype("float32")
    del roll

del fn_sub, grp, prev_delay, prev_time
gc.collect()
print(df[["route_trailing30d_delay", "route_delay_rate_24h", "route_flights_trailing3h",
          "route_flights_trailing24h", "flight_number_prev_delay", "flight_number_prev_gap_days",
          "flight_number_last3_mean_delay", "flight_number_last5_mean_delay",
          "flight_number_last10_mean_delay"]].describe())
```

---

**Cell 10 — Airport Trailing Delay + Rates**

```python
result_cols = [
    "airport_trailing3h_delay", "airport_trailing6h_delay",
    "airport_trailing3h_delay_rate", "airport_trailing6h_delay_rate",
    "airport_delay_trend", "airport_hour_mean_delay_30d", "airport_hour_delay_rate_30d",
]
defaults = {
    "airport_trailing3h_delay": GLOBAL_MEAN_DELAY, "airport_trailing6h_delay": GLOBAL_MEAN_DELAY,
    "airport_trailing3h_delay_rate": GLOBAL_DELAY_RATE, "airport_trailing6h_delay_rate": GLOBAL_DELAY_RATE,
    "airport_delay_trend": 0.0,
    "airport_hour_mean_delay_30d": GLOBAL_MEAN_DELAY, "airport_hour_delay_rate_30d": GLOBAL_DELAY_RATE,
}
for c in result_cols:
    df[c] = np.float32(defaults[c])

airports = df["origin_airport"].dropna().unique()
print(f"Processing {len(airports)} airports...")

for i, airport in enumerate(airports):
    mask = (df["origin_airport"] == airport).values
    idx = df.index[mask]
    if len(idx) == 0:
        continue

    sub = df.loc[idx, ["scheduled_departure", "actual_departure", TARGET]].copy()
    sub["_is_delayed"] = (sub[TARGET] >= DELAY_THRESHOLD_MIN).astype("float32")

    sub["_hbucket"] = sub["actual_departure"].dt.floor("h")
    hourly = (
        sub.groupby("_hbucket")
           .agg(_mean=(TARGET, "mean"), _rate=("_is_delayed", "mean"))
           .reset_index().sort_values("_hbucket")
    )
    hourly["trailing3"] = hourly["_mean"].rolling(3, min_periods=1).mean()
    hourly["trailing6"] = hourly["_mean"].rolling(6, min_periods=1).mean()
    hourly["rate3"] = hourly["_rate"].rolling(3, min_periods=1).mean()
    hourly["rate6"] = hourly["_rate"].rolling(6, min_periods=1).mean()
    hourly["trend"] = hourly["_mean"] - hourly["trailing3"]
    hourly["available_from"] = hourly["_hbucket"] + pd.Timedelta(hours=1)
    hourly = hourly.sort_values("available_from")

    left = sub[["scheduled_departure"]].reset_index().rename(columns={"index": "_orig_idx"})
    left = left.sort_values("scheduled_departure")
    m = pd.merge_asof(
        left, hourly[["available_from", "trailing3", "trailing6", "rate3", "rate6", "trend"]],
        left_on="scheduled_departure", right_on="available_from",
        direction="backward", tolerance=pd.Timedelta(hours=6),
    ).set_index("_orig_idx")

    df.loc[m.index, "airport_trailing3h_delay"] = m["trailing3"].fillna(GLOBAL_MEAN_DELAY).astype("float32")
    df.loc[m.index, "airport_trailing6h_delay"] = m["trailing6"].fillna(GLOBAL_MEAN_DELAY).astype("float32")
    df.loc[m.index, "airport_trailing3h_delay_rate"] = m["rate3"].fillna(GLOBAL_DELAY_RATE).astype("float32")
    df.loc[m.index, "airport_trailing6h_delay_rate"] = m["rate6"].fillna(GLOBAL_DELAY_RATE).astype("float32")
    df.loc[m.index, "airport_delay_trend"] = m["trend"].fillna(0.0).astype("float32")

    sub["_hour"] = sub["scheduled_departure"].dt.hour.astype("int8")
    sub["_date"] = sub["scheduled_departure"].dt.floor("D")
    daily_hr = (
        sub.groupby(["_hour", "_date"])
           .agg(_mean=(TARGET, "mean"), _rate=("_is_delayed", "mean"))
           .reset_index().sort_values(["_hour", "_date"])
    )
    g = daily_hr.groupby("_hour")
    daily_hr["hr_mean_30"] = g["_mean"].transform(lambda s: s.rolling(30, min_periods=3).mean())
    daily_hr["hr_rate_30"] = g["_rate"].transform(lambda s: s.rolling(30, min_periods=3).mean())
    daily_hr["available_from"] = daily_hr["_date"] + pd.Timedelta(days=1)
    daily_hr = daily_hr.sort_values("available_from")

    left2 = sub[["scheduled_departure", "_hour"]].reset_index().rename(columns={"index": "_orig_idx"})
    left2 = left2.sort_values("scheduled_departure")
    m2 = pd.merge_asof(
        left2, daily_hr[["_hour", "available_from", "hr_mean_30", "hr_rate_30"]],
        left_on="scheduled_departure", right_on="available_from",
        by="_hour", direction="backward", tolerance=pd.Timedelta(days=35),
    ).set_index("_orig_idx")

    df.loc[m2.index, "airport_hour_mean_delay_30d"] = m2["hr_mean_30"].fillna(GLOBAL_MEAN_DELAY).astype("float32")
    df.loc[m2.index, "airport_hour_delay_rate_30d"] = m2["hr_rate_30"].fillna(GLOBAL_DELAY_RATE).astype("float32")

    del sub, hourly, left, m, daily_hr, g, left2, m2
    if i % 50 == 0:
        gc.collect()

gc.collect()
print(df[result_cols].describe())
```

---

**Cell 11 — Schedule-Only Structural Features**

```python
df["origin_sched_count_3h"] = trailing_sched_count(df, ["origin_airport"], "scheduled_departure", 3)
df["origin_sched_count_6h"] = trailing_sched_count(df, ["origin_airport"], "scheduled_departure", 6)
_count_24h = trailing_sched_count(df, ["origin_airport"], "scheduled_departure", 24)
df["origin_departure_volume_ratio"] = (
    df["origin_sched_count_3h"] / (_count_24h / 8).replace(0, np.nan)
).fillna(1.0).astype("float32")
del _count_24h
gc.collect()

sched_arrivals = df[["airline", "destination_airport", "scheduled_arrival"]].copy()
sched_arrivals = sched_arrivals.rename(columns={"destination_airport": "airport"})
sched_arrivals = sched_arrivals.sort_values("scheduled_arrival")

sched_deps = df[["airline", "origin_airport", "scheduled_departure"]].copy()
sched_deps = sched_deps.rename(columns={"origin_airport": "airport"})
sched_deps["_orig_idx"] = df.index
sched_deps = sched_deps.sort_values("scheduled_departure")

sched_arrivals["airline"], sched_deps["airline"] = align_categories(sched_arrivals["airline"], sched_deps["airline"])
sched_arrivals["airport"], sched_deps["airport"] = align_categories(sched_arrivals["airport"], sched_deps["airport"])

merged_turn = pd.merge_asof(
    sched_deps, sched_arrivals,
    left_on="scheduled_departure", right_on="scheduled_arrival",
    by=["airline", "airport"], direction="backward", tolerance=pd.Timedelta(hours=6),
).set_index("_orig_idx").sort_index()

turnaround = (merged_turn["scheduled_departure"] - merged_turn["scheduled_arrival"]).dt.total_seconds() / 60
df["scheduled_turnaround_minutes"] = turnaround.reindex(df.index).fillna(999).astype("float32")

del sched_arrivals, sched_deps, merged_turn, turnaround
gc.collect()

bank_sub = df[["airline", "origin_airport", "scheduled_departure"]].copy()
bank_sub["_orig_idx"] = df.index
bank_sub = bank_sub.sort_values(["airline", "origin_airport", "scheduled_departure"])

def bank_density(times, minutes=30):
    t = times.values.astype("datetime64[ns]")
    window = np.timedelta64(minutes, "m")
    lo = np.searchsorted(t, t - window, side="left")
    hi = np.searchsorted(t, t + window, side="right")
    return (hi - lo - 1).astype("float32")

parts = []
for _, g in bank_sub.groupby(["airline", "origin_airport"], observed=True):
    parts.append(pd.Series(bank_density(g["scheduled_departure"]), index=g["_orig_idx"]))
df["connecting_bank_density"] = pd.concat(parts).reindex(df.index).astype("float32")

del bank_sub, parts
gc.collect()
print(df[["origin_sched_count_3h", "origin_sched_count_6h", "origin_departure_volume_ratio",
          "scheduled_turnaround_minutes", "connecting_bank_density"]].describe())
```

---

**Cell 12 — Holiday Flag**

```python
import holidays

country_map = {"United States": "US", "USA": "US", "US": "US"}
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

**Cell 13 — Reattach All Engineered + Weather Features to Splits**

```python
NEW_FEATURES = [
    "origin_hourly_sched_count", "airline_airport_prior_hour_delay", "airport_prior_hour_delay",
    "incoming_aircraft_delay", "has_incoming_match",
    # NEW: aircraft rotation / cascading risk
    "aircraft_leg_number_of_day", "scheduled_gap_since_prev_leg",
    "is_first_flight_of_day_for_tail", "has_rotation_data",
    "route_trailing30d_delay", "route_delay_rate_24h",
    "route_flights_trailing3h", "route_flights_trailing24h",
    "flight_number_prev_delay", "flight_number_prev_gap_days",
    "flight_number_last3_mean_delay", "flight_number_last5_mean_delay", "flight_number_last10_mean_delay",
    "airport_trailing3h_delay", "airport_trailing6h_delay",
    "airport_trailing3h_delay_rate", "airport_trailing6h_delay_rate", "airport_delay_trend",
    "airport_hour_mean_delay_30d", "airport_hour_delay_rate_30d",
    "origin_sched_count_3h", "origin_sched_count_6h", "origin_departure_volume_ratio",
    "scheduled_turnaround_minutes", "connecting_bank_density",
    "is_holiday",
]

for col in NEW_FEATURES + WEATHER_FEATURES:
    train_df[col] = df.loc[train_df.index, col].astype("float32")
    val_df[col] = df.loc[val_df.index, col].astype("float32")
    test_df[col] = df.loc[test_df.index, col].astype("float32")

del df
gc.collect()
print("train:", train_df.shape, " val:", val_df.shape, " test:", test_df.shape)
print("NaN check:", train_df[NEW_FEATURES + WEATHER_FEATURES].isna().sum().sum())
```

---

**Cell 14 — Final Feature Matrices (+ persist to disk for kernel restart)**

```python
FINAL_FEATURES = [
    "airline_te", "origin_airport_te", "destination_airport_te", "month_airport_te", "TAIL_NUM_te",
    "flight_number_freq",
    "scheduled_flight_duration_minutes",
    "hour_of_day_sin", "hour_of_day_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "is_weekend", "is_redeye",
] + WEATHER_FEATURES + NEW_FEATURES

X_train = train_df[FINAL_FEATURES].astype("float32")
y_train = train_df[TARGET].astype("float32")
del train_df
gc.collect()

X_val = val_df[FINAL_FEATURES].astype("float32")
y_val = val_df[TARGET].astype("float32")
del val_df
gc.collect()

X_test = test_df[FINAL_FEATURES].astype("float32")
y_test = test_df[TARGET].astype("float32")
del test_df
gc.collect()

print("train:", X_train.shape, " val:", X_val.shape, " test:", X_test.shape)
print("NaN check:", X_train.isna().sum().sum(), X_val.isna().sum().sum(), X_test.isna().sum().sum())

# --- persist matrices to disk so Cell 15 can run in a fresh, restarted kernel ---
import numpy as np
np.savez(
    "/kaggle/working/matrices.npz",
    X_train=X_train.to_numpy(dtype="float32"), y_train=y_train.to_numpy(dtype="float32"),
    X_val=X_val.to_numpy(dtype="float32"),     y_val=y_val.to_numpy(dtype="float32"),
    X_test=X_test.to_numpy(dtype="float32"),   y_test=y_test.to_numpy(dtype="float32"),
)
# save the column order/names too, so Cell 16's feature_importances_ labeling still works after restart
import json
with open("/kaggle/working/final_features.json", "w") as f:
    json.dump(FINAL_FEATURES, f)

print("Saved matrices.npz + final_features.json to /kaggle/working/")
print("Now: RESTART THE KERNEL, then run Cell 15 fresh (it will load from disk).")
```

---

**Cell 15a — Ridge (chunked closed-form solve, fresh kernel)**

```python
import gc, ctypes, time, json, os
import numpy as np

def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def save_result(name, y_true, y_pred, train_time):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    path = "/kaggle/working/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "train_time_sec": train_time}
    json.dump(results, open(path, "w"))
    print(f"{name} -> RMSE: {rmse:.3f}  MAE: {mae:.3f}  R2: {r2:.4f}  ({train_time:.1f}s)")

# mmap_mode lets us read chunks off disk without loading the whole array into RAM
data = np.load("/kaggle/working/matrices.npz", mmap_mode="r")
X_train_mm = data["X_train"]   # not loaded yet, just a disk-backed view
y_train_mm = data["y_train"]
X_val_np = np.asarray(data["X_val"], dtype=np.float64)
y_val_np = np.asarray(data["y_val"], dtype=np.float64)
n_total, n_features = X_train_mm.shape
print("train (full, chunked):", X_train_mm.shape, " val:", X_val_np.shape)

t0 = time.time()
alpha = 1.0
p = n_features + 1  # +1 for intercept
XtX = np.zeros((p, p), dtype=np.float64)
Xty = np.zeros(p, dtype=np.float64)
chunk_size = 1_000_000

for start in range(0, n_total, chunk_size):
    end = min(start + chunk_size, n_total)
    Xc = np.asarray(X_train_mm[start:end], dtype=np.float64)
    yc = np.asarray(y_train_mm[start:end], dtype=np.float64)
    Xb = np.hstack([Xc, np.ones((Xc.shape[0], 1))])  # bias column
    XtX += Xb.T @ Xb
    Xty += Xb.T @ yc
    del Xc, yc, Xb
    release_memory()
    print(f"  processed {end}/{n_total}")

# Ridge penalty -- don't regularize the intercept term (last row/col)
reg = alpha * np.eye(p)
reg[-1, -1] = 0.0
coef_full = np.linalg.solve(XtX + reg, Xty)
coef, intercept = coef_full[:-1], coef_full[-1]

y_pred = X_val_np @ coef + intercept
save_result("Ridge (full data)", y_val_np, y_pred, time.time() - t0)
```

---

**Cell 15b — Linear Regression (chunked closed-form solve, fresh kernel)**

```python
import gc, ctypes, time, json, os
import numpy as np

def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def save_result(name, y_true, y_pred, train_time):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    path = "/kaggle/working/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "train_time_sec": train_time}
    json.dump(results, open(path, "w"))
    print(f"{name} -> RMSE: {rmse:.3f}  MAE: {mae:.3f}  R2: {r2:.4f}  ({train_time:.1f}s)")

data = np.load("/kaggle/working/matrices.npz", mmap_mode="r")
X_train_mm = data["X_train"]
y_train_mm = data["y_train"]
X_val_np = np.asarray(data["X_val"], dtype=np.float64)
y_val_np = np.asarray(data["y_val"], dtype=np.float64)
n_total, n_features = X_train_mm.shape
print("train (full, chunked):", X_train_mm.shape, " val:", X_val_np.shape)

t0 = time.time()
p = n_features + 1
XtX = np.zeros((p, p), dtype=np.float64)
Xty = np.zeros(p, dtype=np.float64)
chunk_size = 1_000_000

for start in range(0, n_total, chunk_size):
    end = min(start + chunk_size, n_total)
    Xc = np.asarray(X_train_mm[start:end], dtype=np.float64)
    yc = np.asarray(y_train_mm[start:end], dtype=np.float64)
    Xb = np.hstack([Xc, np.ones((Xc.shape[0], 1))])
    XtX += Xb.T @ Xb
    Xty += Xb.T @ yc
    del Xc, yc, Xb
    release_memory()
    print(f"  processed {end}/{n_total}")

coef_full = np.linalg.solve(XtX, Xty)  # no ridge penalty at all
coef, intercept = coef_full[:-1], coef_full[-1]

y_pred = X_val_np @ coef + intercept
save_result("Linear Regression (full data)", y_val_np, y_pred, time.time() - t0)
```

---

**Cell 15c — XGBoost (GPU, fresh kernel)**

```python
import gc, ctypes, time, json, os
import numpy as np

def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def save_result(name, y_true, y_pred, train_time):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    path = "/kaggle/working/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "train_time_sec": train_time}
    json.dump(results, open(path, "w"))
    print(f"{name} -> RMSE: {rmse:.3f}  MAE: {mae:.3f}  R2: {r2:.4f}  ({train_time:.1f}s)")

data = np.load("/kaggle/working/matrices.npz")
X_train_np, y_train_np = data["X_train"], data["y_train"]
X_val_np, y_val_np = data["X_val"], data["y_val"]
data.close(); del data
release_memory()
print("train:", X_train_np.shape, " val:", X_val_np.shape)

import xgboost as xgb
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", device="cuda",
    random_state=42, n_jobs=-1,
)
xgb_model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)], verbose=100)
save_result("XGBoost (GPU)", y_val_np, xgb_model.predict(X_val_np), time.time() - t0)
```

---

**Cell 15d — LightGBM (GPU, fresh kernel)**

```python
import gc, ctypes, time, json, os
import numpy as np

def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def save_result(name, y_true, y_pred, train_time):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    path = "/kaggle/working/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "train_time_sec": train_time}
    json.dump(results, open(path, "w"))
    print(f"{name} -> RMSE: {rmse:.3f}  MAE: {mae:.3f}  R2: {r2:.4f}  ({train_time:.1f}s)")

data = np.load("/kaggle/working/matrices.npz")
X_train_np, y_train_np = data["X_train"], data["y_train"]
X_val_np, y_val_np = data["X_val"], data["y_val"]
data.close(); del data
release_memory()
print("train:", X_train_np.shape, " val:", X_val_np.shape)

import lightgbm as lgb
t0 = time.time()
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    device="gpu", max_bin=63,
    random_state=42, n_jobs=-1,
)
lgb_model.fit(X_train_np, y_train_np, eval_set=[(X_val_np, y_val_np)],
              callbacks=[lgb.log_evaluation(100)])
save_result("LightGBM (GPU)", y_val_np, lgb_model.predict(X_val_np), time.time() - t0)
```

---

**Cell 15e — CatBoost (GPU, fresh kernel) + feature importance dump**

```python
import gc, ctypes, time, json, os
import numpy as np

def release_memory():
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def save_result(name, y_true, y_pred, train_time):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    path = "/kaggle/working/results.json"
    results = json.load(open(path)) if os.path.exists(path) else {}
    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2, "train_time_sec": train_time}
    json.dump(results, open(path, "w"))
    print(f"{name} -> RMSE: {rmse:.3f}  MAE: {mae:.3f}  R2: {r2:.4f}  ({train_time:.1f}s)")

data = np.load("/kaggle/working/matrices.npz")
X_train_np, y_train_np = data["X_train"], data["y_train"]
X_val_np, y_val_np = data["X_val"], data["y_val"]
data.close(); del data
release_memory()
print("train:", X_train_np.shape, " val:", X_val_np.shape)

from catboost import CatBoostRegressor
t0 = time.time()
cat_model = CatBoostRegressor(
    iterations=500, depth=8, learning_rate=0.05,
    task_type="GPU", devices="0:1",
    used_ram_limit="4gb",
    random_state=42, verbose=100,
)
cat_model.fit(X_train_np, y_train_np, eval_set=(X_val_np, y_val_np))
save_result("CatBoost (GPU)", y_val_np, cat_model.predict(X_val_np), time.time() - t0)

# --- feature importance, loaded from disk, no dependency on Cells 1-14 ---
FINAL_FEATURES = json.load(open("/kaggle/working/final_features.json"))
importance_dict = dict(zip(FINAL_FEATURES, cat_model.feature_importances_.tolist()))
json.dump(importance_dict, open("/kaggle/working/feature_importance.json", "w"))
print("Saved feature importance.")
```

---

**Cell 15f — Results comparison table**

```python
import json
import pandas as pd

results = json.load(open("/kaggle/working/results.json"))
comparison = pd.DataFrame(results).T.sort_values("RMSE")
comparison.index.name = "Model"
print(comparison)
```

---

**Cell 16 — Feature Importance Check**

```python
import json
import pandas as pd

importance_dict = json.load(open("/kaggle/working/feature_importance.json"))
importance = pd.Series(importance_dict).sort_values(ascending=False)
print(importance)
```

---

## Results

| Model | RMSE | MAE | R2 | Train time (s) |
|---|---:|---:|---:|---:|
| XGBoost (GPU) | 49.09 | 18.88 | 0.2083 | 136.1 |
| CatBoost (GPU) | 49.44 | 19.13 | 0.1970 | 63.6 |
| LightGBM (GPU) | 49.49 | 19.13 | 0.1954 | 287.3 |
| Linear Regression (full data) | 52.60 | 21.82 | 0.0911 | 7.8 |
| Ridge (full data) | 52.60 | 21.83 | 0.0911 | 7.6 |
| Ridge (sklearn, in-memory) | 52.60 | 21.86 | 0.0910 | 13.7 |

**Takeaways:**
- Gradient-boosted trees (XGBoost/CatBoost/LightGBM) clearly outperform linear models —
  roughly 3 points lower RMSE and more than double the R2 — confirming the engineered
  features (rotation, congestion, trailing-delay) carry non-linear signal the linear
  models can't exploit.
- XGBoost is the current best model on val (lowest RMSE, highest R2), though CatBoost
  trains in under half the time for a very close RMSE — worth keeping both in
  consideration depending on whether inference speed or accuracy is prioritized.
- The two closed-form Ridge/Linear implementations (chunked, full data) and sklearn's
  in-memory `Ridge` land within noise of each other, which is a good sanity check that
  the chunked closed-form solve is mathematically equivalent to the standard fit.
- R2 around 0.20 means there's still a lot of unexplained variance — expected for
  departure delay, which has a large irreducible/random component (weather, ATC,
  mechanical issues not yet captured), but there's likely room for more feature work
  before diminishing returns set in.

---

## Feature Importance (CatBoost)

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | flight_number_prev_delay | 17.55 |
| 2 | flight_number_prev_gap_days | 10.99 |
| 3 | airline_te | 8.35 |
| 4 | incoming_aircraft_delay | 7.30 |
| 5 | scheduled_gap_since_prev_leg | 6.59 |
| 6 | airline_airport_prior_hour_delay | 6.46 |
| 7 | has_incoming_match | 3.36 |
| 8 | airport_hour_mean_delay_30d | 3.29 |
| 9 | scheduled_turnaround_minutes | 3.23 |
| 10 | flight_number_last10_mean_delay | 2.86 |
| 11 | TAIL_NUM_te | 2.58 |
| 12 | airport_trailing3h_delay_rate | 2.46 |
| 13 | flight_number_last3_mean_delay | 2.31 |
| 14 | is_first_flight_of_day_for_tail | 1.95 |
| 15 | hour_of_day_cos | 1.85 |
| 16 | route_trailing30d_delay | 1.76 |
| 17 | connecting_bank_density | 1.58 |
| 18 | hour_of_day_sin | 1.50 |
| 19 | scheduled_flight_duration_minutes | 1.44 |
| 20 | aircraft_leg_number_of_day | 1.36 |
| 21 | airport_prior_hour_delay | 1.35 |
| 22 | origin_sched_count_6h | 0.97 |
| 23 | airport_hour_delay_rate_30d | 0.91 |
| 24 | visibility_m | 0.82 |
| 25 | flight_number_last5_mean_delay | 0.74 |
| 26 | flight_number_freq | 0.66 |
| 27 | destination_airport_te | 0.59 |
| 28 | origin_airport_te | 0.57 |
| 29 | origin_sched_count_3h | 0.44 |
| 30 | origin_departure_volume_ratio | 0.43 |
| 31 | temperature_2m_c | 0.42 |
| 32 | day_of_week_sin | 0.35 |
| 33 | month_cos | 0.32 |
| 34 | airport_trailing6h_delay_rate | 0.31 |
| 35 | airport_trailing6h_delay | 0.29 |
| 36 | is_redeye | 0.29 |
| 37 | month_airport_te | 0.25 |
| 38 | route_delay_rate_24h | 0.22 |
| 39 | route_flights_trailing24h | 0.20 |
| 40 | airport_delay_trend | 0.20 |
| 41 | has_rotation_data | 0.18 |
| 42 | origin_hourly_sched_count | 0.18 |
| 43 | windspeed_10m_ms | 0.14 |
| 44 | cloudcover_low_pct | 0.11 |
| 45 | month_sin | 0.09 |
| 46 | day_of_week_cos | 0.08 |
| 47 | airport_trailing3h_delay | 0.06 |
| 48 | route_flights_trailing3h | 0.04 |
| 49 | is_holiday | 0.01 |
| 50 | is_weekend | 0.004 |

**Notes on the rotation feature group (TAIL_NUM-derived):**
- `scheduled_gap_since_prev_leg` (rank 5) and `is_first_flight_of_day_for_tail` (rank 14)
  landed solidly in the upper-middle of the importance ranking — meaningfully ahead of
  the static airport/route target encodings (`origin_airport_te` rank 28,
  `destination_airport_te` rank 27), confirming same-day turnaround pressure is a real,
  usable signal.
- `TAIL_NUM_te` (rank 11) outperforms `origin_airport_te` and `destination_airport_te`
  individually, supporting the earlier decision to target-encode rather than
  frequency-encode the tail number.
- `aircraft_leg_number_of_day` (rank 20) and `has_rotation_data` (rank 41) contribute
  less on their own — `has_rotation_data` in particular is low-importance, which makes
  sense since it's a fairly rare/binary signal (~1.6% "Unknown" rate) rather than a
  continuously varying feature. Kept in the model for interpretability/completeness
  rather than pure importance.

## Open Items For Next Round
- Destination-side features (weather, congestion, trailing delay) are still missing —
  everything origin-side has a destination-side mirror candidate, per the earlier
  feature brainstorm.
- `cumulative_delay_today_for_tail` and `num_legs_scheduled_today_for_tail` (broader
  rotation features beyond the three built this round) were discussed but not yet
  implemented — the top-of-list performance of `scheduled_gap_since_prev_leg` and
  `flight_number_prev_delay` suggests the cumulative/count versions could add further
  signal.
- R2 ceiling (~0.21) suggests airline-level operational-context features
  (`carrier_recent_OTP_7d`) and slot-controlled-airport flags may be worth prioritizing
  next, per the original five-category feature brainstorm.
