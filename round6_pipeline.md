# Round 6 Pipeline — Flight Delay Prediction

Builds on Round 5. Adds deeper trailing-history features (multi-window flight-number
means, airport 3h/6h trailing delay + rates, airport×hour-of-day 30d stats), schedule-only
structural features (turnaround, connecting bank density, volume ratios), and a
month×airport seasonal target encoding. Rewritten from lettered sub-cells (7b–7j) into a
clean numbered sequence (Cells 1–15) built on a shared helper library (Cell 6).

## Key changes from Round 5
- Feature engineering now goes through shared helpers (`build_bucketed_table`,
  `add_rolling`, `causal_attach`, `trailing_sched_count`) instead of duplicated
  `merge_asof` boilerplate per feature.
- `GLOBAL_MEAN_DELAY` / `GLOBAL_DELAY_RATE` fallback constants now computed from
  **train only**, not the full dataset — previously leaked val/test target distribution
  into the imputation value used for their own missing rows.
- `build_bucketed_table` uses floor + groupby instead of `groupby().resample()`, which
  was materializing a dense grid of every group × every time bucket across the full date
  range (including empty buckets) and caused repeated OOM kernel restarts.
- Cell 9 (airport trailing delay + hour-of-day stats) rewritten as a per-airport loop —
  the vectorized multi-key `merge_asof` plus `df.assign()` (which copies the entire frame)
  was the actual OOM cause there.
- XGBoost prediction switched to `get_booster().inplace_predict()` to fix the GPU/CPU
  device-mismatch warning (was silently falling back to CPU `DMatrix`).
- Each split (`X_train`/`X_val`/`X_test`) is extracted and its source frame (`train_df`/
  `val_df`/`test_df`) is deleted immediately after, rather than holding all three wide
  frames alive until the end of Cell 13.

## Results (n_estimators/iterations = 500)

| Model | RMSE | MAE | R² | Train time (s) |
|---|---|---|---|---|
| XGBoost (GPU) | 50.73 | 20.20 | 0.175 | 252.6 |
| LightGBM (GPU) | 50.83 | 20.38 | 0.172 | 388.4 |
| CatBoost (GPU) | 50.92 | 20.39 | 0.169 | 89.2 |
| Ridge | 53.53 | 22.74 | 0.082 | 8.8 |
| Linear Regression | 53.53 | 22.74 | 0.082 | 25.7 |

Validation samples: 7,834,290. Features: 45.

Ridge continues to throw an `Ill-conditioned matrix` warning — collinearity among the
trailing-delay features (`airport_trailing3h_delay`, `airport_trailing6h_delay`,
`airport_hour_mean_delay_30d`, `airport_delay_trend`, etc.) is the likely cause and is
still unresolved from Round 5. Worth a VIF check before Round 7.

### CatBoost feature importances

```
flight_number_prev_delay            18.14
flight_number_prev_gap_days         12.31
airline_te                          10.80
incoming_aircraft_delay              8.33
airline_airport_prior_hour_delay     7.77
scheduled_turnaround_minutes         4.11
flight_number_last10_mean_delay      3.91
airport_hour_mean_delay_30d          3.84
has_incoming_match                   3.53
airport_trailing3h_delay_rate        3.06
flight_number_last3_mean_delay       2.88
hour_of_day_sin                      2.16
connecting_bank_density              2.13
hour_of_day_cos                      2.06
airport_prior_hour_delay             1.92
route_trailing30d_delay              1.78
scheduled_flight_duration_minutes    1.20
flight_number_last5_mean_delay       0.94
flight_number_freq                   0.77
visibility_m                         0.77
airport_hour_delay_rate_30d          0.77
destination_airport_te               0.77
origin_sched_count_6h                0.73
origin_airport_te                    0.60
origin_departure_volume_ratio        0.54
is_redeye                            0.44
temperature_2m_c                     0.42
origin_sched_count_3h                0.41
airport_trailing6h_delay_rate        0.37
airport_trailing6h_delay             0.37
day_of_week_sin                      0.36
month_cos                            0.31
route_delay_rate_24h                 0.22
month_airport_te                     0.20
airport_delay_trend                  0.19
airport_trailing3h_delay             0.18
windspeed_10m_ms                     0.15
route_flights_trailing24h            0.14
day_of_week_cos                      0.12
origin_hourly_sched_count            0.12
cloudcover_low_pct                   0.08
month_sin                            0.06
route_flights_trailing3h             0.05
is_holiday                           0.01
is_weekend                           0.00
```

**Reading the importances:** the flight-number and incoming-aircraft cluster
(`flight_number_prev_delay`, `flight_number_prev_gap_days`, `incoming_aircraft_delay`,
`airline_airport_prior_hour_delay`) still dominates, same as Round 5. New Round 6
features that earned real weight: `scheduled_turnaround_minutes` (4.11),
`flight_number_last10_mean_delay` (3.91), `airport_hour_mean_delay_30d` (3.84),
`airport_trailing3h_delay_rate` (3.06), `connecting_bank_density` (2.13). Weak/near-zero:
`route_flights_trailing3h`, `is_holiday`, `is_weekend`, `month_sin` — candidates to drop
in a future pruning pass if training time becomes a concern.

---

## Cell 1 — Load Data & Time-Based Split

```python
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
# loading then dropping. Biggest single RAM saving here.
NEEDED_COLS = list(set(
    FEATURES_CAT + FEATURES_NUM + WEATHER_FEATURES +
    [TARGET, TIME_COL, "scheduled_arrival", "actual_arrival", "actual_departure",
     "route", "country"]
))

df = pd.read_parquet("/kaggle/input/datasets/tjaycuz/df-clean-v2", columns=NEEDED_COLS)
df[TIME_COL] = pd.to_datetime(df[TIME_COL])
df["scheduled_arrival"] = pd.to_datetime(df["scheduled_arrival"])
df["actual_arrival"] = pd.to_datetime(df["actual_arrival"])
df["actual_departure"] = pd.to_datetime(df["actual_departure"])

# downcast in place: float64->float32, int64->int32 (halves numeric footprint)
for c in df.select_dtypes(include="float64").columns:
    df[c] = df[c].astype("float32")
for c in df.select_dtypes(include="int64").columns:
    df[c] = df[c].astype("int32")

# repeated-value text columns -> category from the start (small int codes, not full strings)
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

## Cell 2 — Normalize Airport Codes (ICAO -> IATA)

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

## Cell 3 — Target Encoding (airline, origin_airport, destination_airport, month_airport)

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

for col in ["airline", "origin_airport", "destination_airport"]:
    train_df[f"{col}_te"] = kfold_target_encode_train(train_df, col, TARGET, n_splits=5, smoothing=10)
    val_df[f"{col}_te"] = target_encode_from_reference(train_df, val_df, col, TARGET, smoothing=10)
    test_df[f"{col}_te"] = target_encode_from_reference(trainval_df, test_df, col, TARGET, smoothing=10)
gc.collect()

# month x airport seasonal interaction (new in Round 6)
for d in (train_df, val_df, test_df):
    d["month_airport"] = (d["month"].astype(str) + "_" + d["origin_airport"].astype(str)).astype("category")

train_df["month_airport_te"] = kfold_target_encode_train(train_df, "month_airport", TARGET, n_splits=5, smoothing=10)
val_df["month_airport_te"] = target_encode_from_reference(train_df, val_df, "month_airport", TARGET, smoothing=10)
trainval_df = pd.concat([train_df, val_df], axis=0)
test_df["month_airport_te"] = target_encode_from_reference(trainval_df, test_df, "month_airport", TARGET, smoothing=10)
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

del freq_map_train, freq_map_trainval
gc.collect()

del trainval_df
gc.collect()
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

## Cell 6 — Feature Engineering Helpers (shared library)

```python
DELAY_THRESHOLD_MIN = 15

# fallback constants come from TRAIN ONLY - never let val/test target distribution
# leak into the imputation value used for their own missing rows
GLOBAL_MEAN_DELAY = train_df[TARGET].mean()
GLOBAL_DELAY_RATE = (train_df[TARGET] >= DELAY_THRESHOLD_MIN).mean()


def align_categories(s1, s2):
    cats = sorted(set(s1.astype("category").cat.categories) | set(s2.astype("category").cat.categories))
    dtype = pd.CategoricalDtype(categories=cats)
    return s1.astype(dtype), s2.astype(dtype)


def build_bucketed_table(source_df, group_cols, time_col, value_col, freq):
    """
    Bucket value_col into `freq` buckets per group_cols using floor + groupby
    (NOT groupby().resample() -- resample materializes a dense grid of every
    group x every time bucket across the FULL date range, including buckets
    with zero rows, which caused repeated OOM kernel restarts).
    Returns _mean and _rate columns, one row per (group, bucket) that actually occurs.
    """
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
    """
    Attaches feature_cols from `table` onto df_, matched by by_cols via merge_asof(direction='backward')
    on base_time_col. `table` must already encode availability lag in table_time_col
    (e.g. bucket_end + offset) so nothing same-period/future leaks in.
    """
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
    """Pure schedule-based trailing count within `hours`. Schedule is known in advance -> no leakage."""
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

---

## Cell 7 — Airport Congestion + Incoming Aircraft Delay

```python
gc.collect()

# --- origin hourly scheduled count (pure schedule, safe) ---
df["origin_hourly_sched_count"] = trailing_sched_count(df, ["origin_airport"], "scheduled_departure", 1)

# --- airline+airport and airport-only prior-hour delay ---
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

# --- incoming aircraft delay (late-aircraft-arrival proxy) ---
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
df["has_incoming_match"] = (df["incoming_aircraft_delay"] != GLOBAL_MEAN_DELAY).astype("float32")  # approx flag

del arrivals, attach, hourly_a
gc.collect()
print(df[["origin_hourly_sched_count", "airline_airport_prior_hour_delay",
          "airport_prior_hour_delay", "incoming_aircraft_delay"]].describe())
```

---

## Cell 8 — Route & Flight-Number History

```python
# --- route trailing 30d mean delay + trailing counts + 24h delay rate ---
route_daily = build_bucketed_table(df, ["route"], "scheduled_departure", TARGET, "1D")
route_daily = add_rolling(route_daily, ["route"], "_mean", [30], "route_trailing", min_periods=3)
route_daily = add_rolling(route_daily, ["route"], "_rate", [1], "route_delay_rate_")  # yesterday's full-day rate
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

# --- flight-number previous occurrence + last3/5/10 trailing mean ---
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

## Cell 9 — Airport Trailing Delay + Rates (3h/6h, hour-of-day 30d)

Per-airport loop — avoids the dataset-wide multi-key `merge_asof` and `df.assign()`
(which copies the entire frame) that caused OOM in the earlier vectorized version.

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

    # ---- hourly buckets, this airport only ----
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

    # ---- hour-of-day 30-day trailing stats, this airport only ----
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

## Cell 10 — Schedule-Only Structural Features (counts, turnaround, bank density)

```python
# --- origin schedule counts + volume ratio ---
df["origin_sched_count_3h"] = trailing_sched_count(df, ["origin_airport"], "scheduled_departure", 3)
df["origin_sched_count_6h"] = trailing_sched_count(df, ["origin_airport"], "scheduled_departure", 6)
_count_24h = trailing_sched_count(df, ["origin_airport"], "scheduled_departure", 24)
df["origin_departure_volume_ratio"] = (
    df["origin_sched_count_3h"] / (_count_24h / 8).replace(0, np.nan)
).fillna(1.0).astype("float32")
del _count_24h
gc.collect()

# --- scheduled turnaround minutes: gap between incoming flight's SCHEDULED arrival
# and this flight's SCHEDULED departure, same airline+airport (pure schedule, no leakage) ---
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

# --- connecting bank density: same airline+airport flights scheduled within +/-30min ---
bank_sub = df[["airline", "origin_airport", "scheduled_departure"]].copy()
bank_sub["_orig_idx"] = df.index
bank_sub = bank_sub.sort_values(["airline", "origin_airport", "scheduled_departure"])

def bank_density(times, minutes=30):
    t = times.values.astype("datetime64[ns]")
    window = np.timedelta64(minutes, "m")
    lo = np.searchsorted(t, t - window, side="left")
    hi = np.searchsorted(t, t + window, side="right")
    return (hi - lo - 1).astype("float32")  # -1 excludes the flight itself

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

## Cell 11 — Holiday Flag (via country + date)

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

## Cell 12 — Reattach All Engineered + Weather Features to Splits

```python
NEW_FEATURES = [
    "origin_hourly_sched_count", "airline_airport_prior_hour_delay", "airport_prior_hour_delay",
    "incoming_aircraft_delay", "has_incoming_match",
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

## Cell 13 — Final Feature Matrices

```python
FINAL_FEATURES = [
    "airline_te", "origin_airport_te", "destination_airport_te", "month_airport_te",
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
```

---

## Cell 14 — Train All Algorithms (memory-efficient)

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

# Ridge - freed right after eval, not needed downstream
t0 = time.time()
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
evaluate_model("Ridge", y_val, ridge.predict(X_val), time.time() - t0)
del ridge
gc.collect()

# Linear Regression - freed right after eval
t0 = time.time()
lr = LinearRegression()
lr.fit(X_train, y_train)
evaluate_model("Linear Regression", y_val, lr.predict(X_val), time.time() - t0)
del lr
gc.collect()

# XGBoost (GPU) - predict via booster.inplace_predict to avoid the CPU/GPU device-mismatch
# fallback warning, and free the booster after eval
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    tree_method="hist", device="cuda",
    random_state=42, n_jobs=-1,
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
xgb_val_dm = xgb.QuantileDMatrix(X_val, ref=xgb.QuantileDMatrix(X_train))
evaluate_model("XGBoost (GPU)", y_val, xgb_model.get_booster().inplace_predict(X_val), time.time() - t0)
del xgb_model, xgb_val_dm
gc.collect()

# LightGBM (GPU) - freed after eval
t0 = time.time()
lgb_model = lgb.LGBMRegressor(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    device="gpu", random_state=42, n_jobs=-1,
)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.log_evaluation(100)])
evaluate_model("LightGBM (GPU)", y_val, lgb_model.predict(X_val), time.time() - t0)
del lgb_model
gc.collect()

# CatBoost (GPU) - KEPT for Cell 15 feature importance, not deleted here
t0 = time.time()
cat_model = CatBoostRegressor(
    iterations=500, depth=8, learning_rate=0.05,
    task_type="GPU", devices="0:1",
    random_state=42, verbose=100,
)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
evaluate_model("CatBoost (GPU)", y_val, cat_model.predict(X_val), time.time() - t0)
gc.collect()

print("\n" + "=" * 70)
print(pd.DataFrame(results).T.sort_values("RMSE"))
```

---

## Cell 15 — Feature Importance Check

```python
importance = pd.Series(cat_model.feature_importances_, index=FINAL_FEATURES).sort_values(ascending=False)
print(importance)

# cat_model no longer needed after this - free it now
del cat_model
gc.collect()
```

---

## Open items for Round 7
- VIF check on the trailing-delay cluster to address the persistent Ridge
  `Ill-conditioned matrix` warning.
- Consider dropping near-zero-importance features (`route_flights_trailing3h`,
  `is_holiday`, `is_weekend`, `month_sin`) if training time becomes a bottleneck.
- `weather_delta_origin_dest` still not implemented — needs destination-side weather
  columns confirmed in the source parquet first.
- `has_incoming_match` is currently an approximation (checks against the fill value)
  rather than an exact match flag.
