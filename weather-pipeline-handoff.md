# Weather Data Pipeline — Handoff

**Project:** Historical weather per airport-year, for US (BTS) + Brazil (ANAC) departure delay prediction.
**Window:** `2022-01-01` to `2026-05-31`.
**Working dir:** `D:\Departure delay prediction\airport_coordinates`

> ⚠️ **This document is now historical.** Everything below reflects the state *before* the work in "Post-handoff Progress (August 2026)" (see bottom of file). Several items marked "NOT YET RUN" here have since been run and superseded — the pipeline is now **production ready** with a final combined US+Brazil dataset. Jump to that section for current status.

---

## Current Decision: Split source, not one API for everything

| Range | Source | Status |
|---|---|---|
| `2022-01-01` → `2025-08-25` | **NOAA ISD** (bulk file download, no rate limit) | Crosswalk built + validated. Downloader built + parsing spot-checked. **Full download complete, gaps resolved (see below). 4053 / 4056 airport-year combos now exist.** |
| `2025-08-26` → `2026-05-31` | **Open-Meteo** (API, rate-limited) | **Complete — 1049/1049 airports.** Historical Forecast API, `fetch_open_meteo_weather.py`. See "Open-Meteo phase" section for the full story, including a data-loss bug and recovery. |

---

## Open-Meteo phase (2025-08-26 → 2026-05-31) — started this session

**Script:** `fetch_open_meteo_weather.py`, in `airport_coordinates/`. Uses the **Historical Forecast API** (`historical-forecast-api.open-meteo.com`), not the ERA5 Historical Weather API — chosen because it mirrors the live Forecast API's format and has full coverage from 2022 onward, so it lines up cleanly with the ISD cutoff.

**Fields fetched (8, matching the approved feature list):**

| Existing `weather_cache` column | Open-Meteo source field | Notes |
|---|---|---|
| `temperature_2m_c` | `temperature_2m` | °C, default unit |
| `windspeed_10m_ms` | `windspeed_10m` | forced `windspeed_unit=ms` (API defaults to km/h) |
| `windgusts_10m_ms` | `windgusts_10m` | same unit fix |
| `visibility_m` | `visibility` | meters, default |
| `precipitation_mm` | `precipitation` | mm, default |
| `snowfall_cm` | `snow_depth` | **not** Open-Meteo's `snowfall` var — mapped from `snow_depth` (actual standing depth) to match what the existing ISD `snowfall_cm` column really holds per the parser docstring caveat. Converted m→cm (API returns depth in meters). ⚠️ Open-Meteo's default `best_match` model blend (IFS) doesn't provide `snow_depth` at all for some regions/dates — expect genuine NaNs here, not just "no snow." Not yet QC'd. |
| `cloudcover_low_pct` | `cloudcover_low` | unchanged |
| `weathercode` | `weathercode` | kept raw, same as ISD — still needs the ISD↔Open-Meteo mapping (open item, unchanged) |

**Output:** writes directly into the same `weather_cache/` used by ISD, same filename convention (`{airport_code}_{year}.parquet`, split at the year boundary → a `_2025` partial-year file + a `_2026` partial-year file per airport), **same 12-column schema as existing ISD parquets** — confirmed column-for-column against a real ISD file before running. No `airport_code` column (airport identity lives in filename only, matching ISD convention). Tracking columns (`weather_station_id`, `is_substitute_station`, `station_distance_km`) are set to `"open-meteo"` / `0` / `0.0` on every row for schema parity with `add_station_tracking_columns.py`'s output — **not yet verified this is what that script expects downstream**, worth a check before the final merge.

**Rate limiting / resilience:**
- Free tier: 600 calls/min, 5,000/hour, 10,000/day, 300,000/month. A single-airport request (~9 months, 8 fields) costs multiple "call units" internally (Open-Meteo bills by `(vars/10) × (days/14)`), so the full 1049-airport run may exceed the 10k/day budget in one calendar day.
- Throttled to ~1 request per 1.5s deliberately (bottleneck is actually server response latency + timeout retries, not this delay — reducing it wouldn't meaningfully speed things up).
- **Checkpointed** (`openmeteo_checkpoint.json`) — safe to stop/restart anytime; already-completed airports are skipped automatically. If the daily call budget is hit overnight, expect a burst of 429s and a stall; just re-run the script again (same command) and it resumes.
- Individual request timeouts (60s) are common and retried automatically with backoff — seen regularly in the log (ABR, ABY, ACT, AEX, etc.), all recovered on retry. Not a concern unless they start failing after all retries.

**Status: complete.** All 1049 airports fetched. See below for a data-loss bug that happened mid-run, how it was recovered, and a duplicate-timestamp issue it surfaced.

---

## ⚠️ Data-loss bug + recovery (2026-07-30)

**What happened:** `fetch_open_meteo_weather.py` and `download_isd_weather.py` both write to the same filename for the 2025 file (`{airport}_2025.parquet` — ISD covers Jan–Aug, Open-Meteo covers Aug–Dec, same file). The first version of the Open-Meteo script wrote with unconditional overwrite (`to_parquet()`), same flaw already documented once before for the ISD re-run (see item 2 in "Progress since last handoff" below). Result: **every airport's `_2025.parquet` file lost its Jan–Aug ISD data**, replaced with only the Aug26–Dec31 Open-Meteo slice, as each airport was processed during the ~10-hour Open-Meteo run.

**Fix:** `write_year_parquet()` was added to `fetch_open_meteo_weather.py` — reads any existing file, concats with new data, dedupes on `datetime`, sorts, writes. No more silent overwrite.

**Recovery sequence used (in order):**
1. `python download_isd_weather.py` — regenerated ISD data from `isd_raw_cache/` (already downloaded, no re-download needed), restoring the lost Jan–Aug 2025 data.
2. `python add_station_tracking_columns.py` — restores tracking columns wiped by step 1 (per the existing documented rule).
3. **`reprocess_from_raw_cache.py`** (new script) — re-wrote parquets from the **already-fetched** Open-Meteo raw JSON cached in `openmeteo_raw_cache/{code}.json` during the original run, using the fixed merge-safe writer. **No API re-calls needed** — recovered 1048/1049 airports from cache in under a minute, avoiding a full ~10-hour re-fetch.
4. Only 1 airport (`LEVT`, Vitoria, Spain — ICAO code, part of an international route in the Brazil dataset) needed a real re-fetch — it was timing out on the full 9-month single request. Fixed via `fix_levt.py`, which split the request into two smaller date-range chunks with a longer timeout (120s vs 60s). Succeeded.

**End state: all 1049 `_2025.parquet` files now correctly contain ISD (Jan–Aug) + Open-Meteo (Aug–Dec) merged into one file.** 2022–2024 and 2026 files were never affected by this bug (different filenames, no collision).

---

## ⚠️ Duplicate-timestamp issue — now quantified, decision made

The "still open" duplicate-timestamp item (see item 8 below) turned out to be much bigger than a footnote. While fixing the above, discovered:

- **2,342 / 5,137 files (~46%) have duplicate-timestamp rows** — concentrated entirely in 2022–2024 (pure ISD years). Some airports lose **30–48% of rows to duplication** (`SABE` Buenos Aires worst at 47.7%).
- Checked whether duplicates are exact copies or genuinely conflicting values: **~89% have different values at the same timestamp** (e.g. two `SABE 2022-01-01 00:00` rows, one 25.7°C, one 27.0°C) — this is NOAA ISD reporting multiple observation types (routine + "special") in the same hour. Not safe to just drop one arbitrarily.
- **Decision made: average numeric columns, most-common value for `weathercode`, per duplicate datetime group.**
- ⚠️ **Known inconsistency, not yet fully resolved:** during the recovery above, the merge-safe writer's `drop_duplicates(keep=...)` already silently deduped 2025 files with an arbitrary keep-first/keep-last rule — **not** the average-based decision above. `dedup_and_remerge_weather.py` was written to fix this properly across **all** years (2022–2026) consistently: regenerates ISD, applies the averaging rule to 2022–2024 and to the ISD portion of 2025, then re-merges cached Open-Meteo data back in (again from `openmeteo_raw_cache/`, no API calls). **This script was provided but NOT YET RUN — do this before trusting any weather data downstream.**

---

## Feature-file merge — script ready, NOT YET RUN

Inspected `us_features_v1.csv` (30,538,520 rows) and `brazil_features_v1.csv` (4,270,516 rows). Key facts for the merge:
- Join key: `origin_airport` (US: IATA e.g. `ATL`; Brazil: ICAO e.g. `SBBE`, matches `weather_cache` filenames either way) + the hour of `scheduled_departure`, floored to `:00`.
- `hour_of_day` / `day_of_week` / `month` columns already in the features files are plain integers, not timestamps (unrelated to the weather merge).
- **Decision made, not yet challenged:** joining origin-airport weather only (departure-side), at scheduled departure hour. Destination-side weather (for `arrival_delay_minutes` modeling) is not yet joined — easy to add later the same way, using `destination_airport` + `scheduled_arrival`.
- `merge_weather_into_features.py` — written, loads weather only for the airports actually present in each country's file (not all 1049, to keep memory sane), left-joins so no flight rows are dropped, prefixes weather columns `origin_wx_*`, saves to a **new** file (`*_with_weather.parquet`) rather than overwriting the original. **Run once per country** (`COUNTRY = "US"` or `"Brazil"` at the top of the script).
- **NOT YET RUN.** Do this only after `dedup_and_remerge_weather.py` has been run (previous section) — otherwise the merge will pull in weather data with the un-decided arbitrary dedup still baked into the 2025 files.

---

## Progress since last handoff

1. **`airport_station_crosswalk.csv` — done, unchanged.** 1014 / 1049 airports matched to a NOAA ISD station with full project-window coverage. 96.7% match rate. (See prior handoff for the 35 unmatched / out-of-scope airports — not touched in this pass.)

2. **`download_isd_weather.py` — full run complete, including a re-run to resolve retriable gaps.**
   - First full run: 905 unique stations, 4042 / 4056 expected airport-year parquets written, 14 gaps.
   - Diagnosed all 14 gaps (`diagnose_missing_gaps.py`) by cross-referencing the crosswalk, `download_isd_weather.log`, and raw cache state. Found the 14 gaps were really only **11 distinct station-year fetches** (a few gap airports share a nearest station):
     - **5 station-years were real timeouts** (retriable): `87347099999`/2024, `87121099999`/2024, `86678099999`/2023 (serves SDLO+SNCL+SNVB), `86942099999`/2023, `96749099999`/2022 (WIII).
     - **6 station-years were confirmed 404s** (not retriable — station genuinely has no file for that specific year): `41599099999`/2024 (OPST), `86622099999`/2022 (serves SBJI+SSKW), `81784099999`/2023 (SBTU), `85921199999`/2022 (SCCF), `84071099999`/2022 (SEQU), `86919099999`/2022 (SSKM).
   - **Re-ran `download_isd_weather.py`** — all 5 timeout station-years resolved successfully on retry. **SACE 2024, SANT 2024, SDLO 2023, SNCL 2023, SNVB 2023, SSUV 2023, WIII 2022 are now in `weather_cache/`.**
   - ⚠️ **Operational gotcha discovered:** `download_isd_weather.py` writes every parquet unconditionally whenever it has data for that station-year — it does not check whether the file already exists or already has extra columns. So this re-run **silently overwrote all ~4042 previously-tagged "normal" parquets**, wiping the tracking columns added earlier that same day (see item 3). **Rule going forward: always re-run `add_station_tracking_columns.py` immediately after any `download_isd_weather.py` re-run.**

3. **Alternate-station fallback built for the 6 confirmed-404 station-years (`fill_alternate_stations.py`).**
   - For each of the 7 affected airport-years, searched the 5 nearest ISD stations (excluding the airport's normal crosswalk station), verified real file existence for that specific missing year (not metadata), and parsed with the same functions as `download_isd_weather.py`.
   - **All 7 found a working alternate station.** Results in `alternate_station_fill_results.csv`.
   - **Decision: kept 4, reverted 3** (distance too far to trust — over the original 100km crosswalk cutoff):

     | Airport | Year | Alternate station | Distance | Outcome |
     |---|---|---|---|---|
     | SCCF | 2022 | 85432099999 | 0.42 km | **Kept** |
     | SEQU | 2022 | 84072599999 | 13.9 km | **Kept** |
     | OPST | 2024 | 41600099999 | 16.3 km | **Kept** |
     | SSKM | 2022 | 83767099999 | 68.7 km | **Kept** |
     | SBTU | 2023 | 82562099999 | 187.5 km | **Reverted** (`revert_far_substitutes.py`) — back to a genuine gap |
     | SSKW | 2022 | 86642099999 | 197.3 km | **Reverted** — back to a genuine gap |
     | SBJI | 2022 | 86642099999 | 277.5 km | **Reverted** — back to a genuine gap |
   - Reverted rows are kept in `alternate_station_fill_results.csv` (not deleted) with `kept=False`, so the record of what was tried and why it was rejected survives.

4. **Tracking columns added to every `weather_cache` parquet (`add_station_tracking_columns.py`).**
   - Adds `weather_station_id`, `is_substitute_station` (0/1), `station_distance_km` (0 for normal, real distance for substitutes) as constant columns on every row, so this metadata survives the later merge into `us_features_v1` / `brazil_features_v1` without needing a separate join.
   - Script is idempotent (drops + re-adds fresh each run) and filters substitutes on `found == True AND kept == True`, so a reverted airport-year can never get mistakenly re-tagged with rejected substitute data even if its original station comes back online in some future re-run.
   - **Currently applied and verified correct** across all 4053 existing files (confirmed via `check_weather_cache_status.py`).

5. **Final gap count: 3 genuine, accepted gaps.** `SBJI 2022`, `SBTU 2023`, `SSKW 2022` have no weather data and are not expected to get any — the nearest trustworthy alternate station was too far away. This is a **deliberate, documented decision**, not an oversight. (0.07% of the full 4056-combo dataset.)

6. **Cleanup recommended, not yet done:** delete `__pycache__`, `test_5_stations.py`, `missing_gaps_diagnosis.csv` (superseded by `alternate_station_fill_results.csv`); archive the `*.log` files and `probe_rate_limits.py` to a subfolder rather than deleting.

7. **⚠️ `weather_cache_qc_summary.csv` is now stale** — generated before the substitute-station work and the timeout re-run, so it doesn't reflect current `weather_cache/` contents. Re-run the QC cell before trusting it again.

8. **Still open from before, not addressed in this pass:**
   - Duplicate-timestamp dedup decision (see prior handoff — NOAA's raw files can have >1 row per timestamp; not yet resolved, needs to happen before the ISD+Open-Meteo merge).
   - Brazil-station spot-check for the `2025-08-25` ISD cutoff (only 1 US-ish station has been checked so far).
   - `weathercode` (MW1) → common-schema mapping between ISD and Open-Meteo codes.
   - `cloudcover_low_pct` (GA1) layer-ordering assumption still not independently cross-checked.

---

## Files (all in `/mnt/user-data/outputs/` from chat, copy to working dir)

| File | Purpose | Status |
|---|---|---|
| *(all files from prior handoff, unchanged — see there)* | | |
| `diagnose_missing_gaps.py` | Classifies each missing airport-year gap as 404 / timeout / parse-failure / never-attempted | **Done, one-time diagnostic, keep for reference** |
| `fill_alternate_stations.py` | Finds + downloads + parses an alternate ISD station for confirmed-404 gaps | **Done, ran once, produced `alternate_station_fill_results.csv`** |
| `alternate_station_fill_results.csv` | Record of all 7 substitute attempts, `kept` column marks final decision | **Done** |
| `revert_far_substitutes.py` | Deletes the 3 too-far substitute parquets, marks them `kept=False` | **Done, ran once** |
| `add_station_tracking_columns.py` | Adds `weather_station_id` / `is_substitute_station` / `station_distance_km` to every parquet | **Done, patched to respect `kept`, re-run after every `download_isd_weather.py` re-run** |
| `check_weather_cache_status.py` | Confirms gap count + tracking-column health across `weather_cache/` | **Done, use anytime to re-verify state** |
| `fetch_open_meteo_weather.py` | Fetches Open-Meteo Historical Forecast data for `2025-08-26`→`2026-05-31`, merge-safe write (fixed after the overwrite bug) | **Done, all 1049 airports fetched, script is now safe for future re-runs** |
| `reprocess_from_raw_cache.py` | Re-writes parquets from already-fetched `openmeteo_raw_cache/*.json` without re-calling the API | **Done, used once for recovery, keep for reference / future recovery scenarios** |
| `fix_levt.py` | One-off: re-fetches a single airport with a smaller date-chunk + longer timeout | **Done, one-time fix for LEVT specifically** |
| `audit_duplicate_timestamps.py` | Quantifies duplicate-timestamp rows across all `weather_cache` files | **Done, produced `duplicate_timestamp_audit.csv`** |
| `check_duplicate_nature.py` | Checks whether duplicate rows are exact copies or conflicting values | **Done, confirmed ~89% are genuinely conflicting, not safe to drop blindly** |
| `dedup_and_remerge_weather.py` | Applies the average/mode dedup decision across all years, re-merges cached Open-Meteo into 2025 | ~~Written, NOT YET RUN~~ **Superseded — dedup approach re-implemented via hourly groupby(mean/mode/first) directly in the new merge pipeline; see Post-handoff Progress §1–2** |
| `merge_weather_into_features.py` | Joins `weather_cache` into `us_features_v1.csv` / `brazil_features_v1.csv` on origin airport + departure hour | ~~Written, NOT YET RUN~~ **Superseded — replaced by a new chunked/streaming production merge pipeline; see Post-handoff Progress §4** |

---

## Immediate next steps — STATUS AS OF ORIGINAL HANDOFF (superseded, kept for history)

1. ~~Run `dedup_and_remerge_weather.py`~~ — **DONE**, via the new production pipeline's hourly aggregation. See Post-handoff Progress §1–2.
2. ~~Run `merge_weather_into_features.py`~~ — **DONE**, via the new streaming merge pipeline, for both countries. See Post-handoff Progress §4, §7.
3. QC the Open-Meteo `snowfall_cm` coverage — **still open.** Final coverage numbers now exist (US 19.63%, Brazil 17.64% snow depth — see Post-handoff Progress final coverage table) but the specific "all-NaN due to missing IFS `snow_depth`" root-cause check has not been separately confirmed.
4. Decide whether to add **destination-side weather** — **still open.** Final dataset is origin-only, as originally decided.
5. Spot-check 2-3 more ISD stations incl. a Brazil one for the `2025-08-25` cutoff — **still open**, not addressed in the post-handoff work.
6. Build the `weathercode` → common-schema mapping — **still open.** `weather_code` coverage is notably low in the final dataset (US 24.86%, Brazil 29.71%) partly because of this unresolved mapping — worth revisiting.
7. Re-run the QC cell to refresh `weather_cache_qc_summary.csv` — **superseded.** Final QC was done directly on the combined output (see Post-handoff Progress §12) rather than on the cache-level summary file.
8. Recommended cleanup — **still open**, not addressed in the post-handoff work.

**New open item surfaced in post-handoff work:** 21 duplicate rows out of 4.27M in the Brazil portion of the final combined dataset need inspection (true duplicates vs. duplicated flights vs. merge artefacts) — see Post-handoff Progress §13.

---

## Post-handoff Progress (August 2026)

Work completed in this session that supersedes the "NOT YET RUN" items above. **The pipeline is now production ready.**

### 1. NOAA duplicate-timestamp problem fully resolved (US)

Auditing the weather cache confirmed many NOAA ISD files had multiple observations within the same hour (e.g. a routine + a "special" observation both at `2022-01-01 03:00`) — genuinely different reports, not file corruption. Because weather was joined hourly, this caused one weather hour to match multiple weather rows, exploding row counts and degrading merge quality.

**Example (JFK):** 13,324 original rows → 8,760 unique hourly timestamps → 4,564 duplicate hourly timestamps (nearly half the rows were duplicates).

**Fix — hourly aggregation, per airport/year, grouped by hour:**
- Numeric weather columns → `mean()`
- Weather code → `mode()`
- Station metadata → `first()`

Result: every airport-year has exactly one row per hour. JFK went from 13,324 rows → 8,760 rows.

### 2. Entire US weather cache repaired

Only JFK was repaired first, to validate the approach. Once confirmed, **all 2,866 US airport weather files were repaired.** Final verification: **0** US airport files still contain duplicate timestamps — every US airport-year now has unique hourly timestamps and is safe for a one-to-one join.

### 3. Root cause of poor weather coverage found

Coverage had looked terrible (~25%) and was suspected to be a `pandas` merge bug. It wasn't — the merge logic was correct. The cause was the duplicate hourly weather rows above, causing incorrect joins. After repairing the cache, JFK coverage went to **100%**.

### 4. Weather merge completely rewritten

The merge script referenced in the original handoff (`merge_weather_into_features.py`) was never actually used. A new production merge pipeline was written instead, driven by scale problems (30M+ rows, pandas memory pressure, schema inconsistency, Parquet-writing issues). New pipeline: chunked CSV reading, weather loaded once, streaming merge, `ParquetWriter` output, Snappy compression. **This is now the production merge pipeline.**

### 5. Fixed Parquet schema mismatch

Streaming writes hit a `Table schema does not match` error — some chunks inferred `flight_number` as `int64`, later chunks as `double`, and PyArrow refuses schema changes mid-write. Fixed with explicit dtype normalization before writing, which stabilized all chunk writes.

### 6. Fixed Windows file-locking issues

Recurring `PermissionError: WinError 32`, caused by the notebook / parquet reader / writer holding open file handles. Resolved by restarting the kernel, clearing outputs, deleting the partial file, and rerunning — pure Windows file-lock behavior, no code change needed.

### 7. US weather merge completed

**Output:** `us_features_v1_with_weather.parquet` — **30,538,520 rows**, coverage **99.84%** (temperature), **99.85%** (visibility) — exceeding original expectations.

### 8. Brazil weather merge rebuilt

The Brazil weather parquet was accidentally lost mid-work. Rather than attempting recovery, the entire merge was regenerated from source. **Output:** `brazil_features_v1_with_weather.parquet`, coverage **92.88%**.

### 9. Added `weather_hour` to Brazil

US had a `weather_hour` column that Brazil lacked (US: 35 columns vs. Brazil: 34). Fixed before merging countries — Brazil now includes `weather_hour` and the schemas match.

### 10. Final US/Brazil schema verification

Confirmed identical columns, names, ordering, and datatypes between US and Brazil — ready for `UNION ALL` with no transformations needed.

### 11. Final combined dataset created

**Output:** `flight_delay_dataset_2022_2026_US_Brazil_with_weather.parquet`
**Location:** `D:\Departure delay prediction\`
**Rows:** 34,809,036 · **Columns:** 35 · **Countries:** US, Brazil

### 12. Final QC performed

Verified row count (34,809,036), both countries present, flight status distribution (Completed / Cancelled / Diverted / Unknown), and weather columns.

### 13. Brazil duplicate inspection (open item)

Detected **21 duplicate rows out of 4.27 million** in Brazil (0.00049%). Still needs inspection to determine whether they're true duplicates, duplicated flights, or merge artefacts — **no evidence yet that this affects model quality**, but not formally closed out.

### Final weather coverage

**United States**

| Feature | Coverage |
|---|---|
| Temperature | 99.84% |
| Wind Speed | 99.81% |
| Wind Gust | 30.11% |
| Visibility | 99.85% |
| Precipitation | 97.07% |
| Snow Depth | 19.63% |
| Cloud Cover | 97.91% |
| Weather Code | 24.86% |

**Brazil**

| Feature | Coverage |
|---|---|
| Temperature | 92.88% |
| Wind Speed | 92.87% |
| Wind Gust | 19.64% |
| Visibility | 91.27% |
| Precipitation | 24.40% |
| Snow Depth | 17.64% |
| Cloud Cover | 68.25% |
| Weather Code | 29.71% |

### Finalized design decisions

- **Weather source:** NOAA ISD for `2022-01-01`→`2025-08-25`; Open-Meteo for `2025-08-26`→`2026-05-31`.
- **Join strategy:** origin airport + scheduled departure hour. Departure-side weather only (destination-side still an open item).
- **Weather aggregation for duplicate hourly observations:** mean (numeric), mode (`weathercode`), first (station metadata).
- **Output format:** Parquet, Snappy compression.
- **Final dataset:** 34,809,036 rows, 35 columns, US + Brazil, weather merged. **Production ready.**
