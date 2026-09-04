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

## Pipeline Cells

**Cell 1 — Load Data & Time-Based Split**
Loads the corrected 2022-2025 parquet, downcasts to `float32`/`int32`, categorizes
`TAIL_NUM` alongside the other categoricals, splits by full calendar year (2022-2023 /
2024 / 2025). `trainval_df` built once here; later cells update it column-by-column
instead of re-concatenating the whole frame.

**Cell 2 — Normalize Airport Codes (ICAO -> IATA)**
Unchanged from Round 6. `trainval_df` updated via single-column concat rather than a
full rebuild.

**Cell 3 — Target Encoding**
`airline`, `origin_airport`, `destination_airport`, `month_airport` (unchanged) +
**`TAIL_NUM`** (new). Same out-of-fold + smoothing=10 methodology throughout.

**Cell 4 — Frequency Encoding (`flight_number`)**
Unchanged. `trainval_df` deleted immediately after, since nothing downstream needs it.

**Cell 5 — Cyclical Encoding**
Unchanged (`hour_of_day`, `day_of_week`, `month` -> sin/cos pairs).

**Cell 6 — Feature Engineering Helpers**
Unchanged (`align_categories`, `build_bucketed_table`, `add_rolling`, `causal_attach`,
`trailing_sched_count`).

**Cell 7 — Airport Congestion + Incoming Aircraft Delay**
Unchanged from Round 6.

**Cell 8 — NEW: Aircraft Rotation / Cascading Risk Features**
Pure schedule-based, computed once on the full `df` (no target values touched, so no
leakage risk from computing before the split-specific stats). Sorts by
`TAIL_NUM` + `scheduled_departure`, groups by tail+day:
- `aircraft_leg_number_of_day` — this leg's position in the tail's rotation that day
- `scheduled_gap_since_prev_leg` — minutes since that tail's previous scheduled leg
  today (-1 sentinel if first leg)
- `is_first_flight_of_day_for_tail` — boolean flag
- `has_rotation_data` — 0 for the `"Unknown"` tail bucket, which gets neutral fallback
  values (leg 1, gap -1, is-first=1) instead of a fabricated sequence

**Cell 9 — Route & Flight-Number History**
Unchanged from Round 6.

**Cell 10 — Airport Trailing Delay + Rates**
Unchanged from Round 6 (per-airport loop building hourly/daily trailing delay stats via
`merge_asof`).

**Cell 11 — Schedule-Only Structural Features**
Unchanged (turnaround minutes, connecting bank density, departure volume ratio).

**Cell 12 — Holiday Flag**
Unchanged.

**Cell 13 — Reattach All Engineered + Weather Features to Splits**
`NEW_FEATURES` list extended with the four Cell 8 rotation features.

**Cell 14 — Final Feature Matrices**
`FINAL_FEATURES` extended with `TAIL_NUM_te` and the rotation features. New this round:
matrices are converted to plain numpy arrays and saved to
`/kaggle/working/matrices.npz`, with `FINAL_FEATURES` order saved to
`final_features.json` — enabling the kernel restart before model training.

**Cell 15 (multi-part, each in its own fresh-kernel cell) — Train All Algorithms**
- Ridge: chunked closed-form solve (`(XtX + alpha*I)^-1 Xty`) over 1M-row chunks from
  the memory-mapped `.npz`, intercept excluded from regularization.
- Linear Regression: same chunked approach, no ridge penalty.
- XGBoost (GPU): `tree_method="hist"`, `device="cuda"`.
- LightGBM (GPU): `device="gpu"`, `max_bin=63` (reduced from default to control GPU
  memory).
- CatBoost (GPU): `task_type="GPU"`, `used_ram_limit="4gb"`.
- Each cell loads fresh from disk, trains, saves its result to `results.json`, and can
  be run after a kernel restart independent of the others.

**Cell 16 — Feature Importance Check**
Loaded from `feature_importance.json` (CatBoost), no dependency on any earlier cell's
live objects.

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
