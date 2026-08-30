# Tail Number Merge — Documentation

## Goal
Attach aircraft **tail numbers** to the main flight delay dataset by matching each flight
against separate US and Brazil tail-number source files, for years **2022–2025 only**
(2026 source files exist but were deliberately excluded from the lookup).

- Input: `df_clean_v2.parquet`
- Output: `df_final.parquet` (same rows/columns as input, plus one new column: `TAIL_NUM`)

---

## Source Data

### US tail number files
- Location: `D:\Departure delay prediction\tail_number_us`
- One zip per month (e.g. `JAN 2022.zip`), each containing `T_ONTIME_MARKETING.csv`
- Columns used: `FL_DATE, OP_UNIQUE_CARRIER, TAIL_NUM, OP_CARRIER_FL_NUM, ORIGIN, DEST`
- `FL_DATE` format: `M/D/YYYY H:MM:SS AM/PM` (e.g. `4/1/2022 12:00:00 AM`) — required
  explicit `format='%m/%d/%Y %I:%M:%S %p'` in `pd.to_datetime` to avoid slow/ambiguous
  parsing.
- Flight numbers are **not** zero-padded (matches df1 after normal string casting).

### Brazil tail number files
- Location: `D:\Departure delay prediction\tail_number_brazil`
- One CSV per month (e.g. `Movimentacoes_Aeroportuarias_202201.csv`)
- Key columns: `NR_MOVIMENTO_TIPO` (`'D'`=departure, `'P'`=landing/arrival),
  `NR_AERONAVE_MARCAS` (tail number), `NR_AERONAVE_OPERADOR` (airline),
  `NR_AEROPORTO_REFERENCIA` (the Brazilian airport the file is anchored to),
  `NR_VOO_OUTRO_AEROPORTO` (the other airport, foreign or domestic),
  `NR_VOO_NUMERO` (flight number), `DT_PREVISTO` (scheduled date)

**Quirks found and handled:**
1. Every file except `Movimentacoes_Aeroportuarias_202207.csv` has a junk first line
   (`"Atualizado em: ..."`) before the real header — detected dynamically per file by
   checking whether `NR_AERONAVE_MARCAS` appears on line 0.
2. Column names have a BOM character glued onto the first column
   (`ï»¿NR_AERONAVE_MARCAS`) — stripped from column names before use.
3. `NR_VOO_NUMERO` is **not** zero-padded, unlike df1's `flight_number` — fixed by
   stripping leading zeros from both sides before building the merge key
   (`.str.lstrip('0')`, with empty string replaced by `'0'`).
4. **Critical fix:** the Brazil file only logs movements *at Brazilian airports*,
   referenced from Brazil's side. An inbound international flight (foreign origin →
   Brazil destination) is logged as `NR_MOVIMENTO_TIPO='P'` with
   `NR_AEROPORTO_REFERENCIA` = the Brazilian airport (which is actually the
   *destination*) and `NR_VOO_OUTRO_AEROPORTO` = the foreign origin. Using only `'D'`
   rows missed all international arrivals into Brazil. Fixed by building merge keys from
   **both** `'D'` and `'P'` rows, with origin/destination correctly reordered for each
   case (see Merge Key Strategy below).

---

## Main Dataset (`df1`) Relevant Columns
```
airline, flight_number, origin_airport, destination_airport,
scheduled_departure, country (values: 'US', 'Brazil')
```
- `flight_number` in df1 is **zero-padded** (e.g. `"0904"`)
- `origin_airport` / `destination_airport` mix IATA (US) and ICAO (Brazil / int'l legs)

---

## Merge Key Strategy
Match on a concatenated string: `airline_code + flight_number + origin_airport + destination_airport + flight_date`

### US
```python
merge_key = OP_UNIQUE_CARRIER + "_" + OP_CARRIER_FL_NUM + "_" + ORIGIN + "_" + DEST + "_" + FL_DATE
```

### Brazil — departures (`NR_MOVIMENTO_TIPO == 'D'`), Brazil airport = origin
```python
merge_key = NR_AERONAVE_OPERADOR + "_" + NR_VOO_NUMERO_clean + "_" +
            NR_AEROPORTO_REFERENCIA + "_" + NR_VOO_OUTRO_AEROPORTO + "_" + DT_PREVISTO
```

### Brazil — arrivals (`NR_MOVIMENTO_TIPO == 'P'`), Brazil airport = destination
```python
merge_key = NR_AERONAVE_OPERADOR + "_" + NR_VOO_NUMERO_clean + "_" +
            NR_VOO_OUTRO_AEROPORTO + "_" + NR_AEROPORTO_REFERENCIA + "_" + DT_PREVISTO
```
(Origin/destination order flipped vs. the departure case, so the key lines up with how
df1 represents the same flight: foreign origin → Brazilian destination.)

Both Brazil key sets are combined into one lookup, deduplicated on `merge_key`
(`keep='first'`).

---

## Match Rate Progress
| Stage | US match % | Brazil match % |
|---|---|---|
| Initial (D-only, no flight-number fix) | 90.62% | 70.06% |
| After stripping leading zeros from flight numbers | 90.62% | 73.23% |
| After including 'P' (arrival) rows for inbound intl flights | 90.62% | 86.00% |

**Final numbers:**
- US matched: 27,205,150 / 30,021,935 (90.62%)
- Brazil matched: 3,409,151 / 3,964,320 (86.00%)

**Known remaining gap (~14%, Brazil):** not fully root-caused. Suspected causes include
flight-number suffix variants (e.g. `950P`) present in the Brazil source but not in df1,
and some genuinely missing records in the Brazil source itself (spot-checked at least one
case with zero matching row in the source on any date/movement type). Considered an
acceptable match rate; not pursued further.

---

## Memory-Efficiency Notes
The naive approach (build full lookup DataFrames, then `pd.read_parquet()` the entire
30M+ row main dataset, then `.merge()`) repeatedly hit `ArrowMemoryError: malloc failed`
on this machine, because the full main dataset plus both lookup tables didn't fit in RAM
simultaneously.

**Final approach — streaming merge, one batch at a time:**
1. Build `us_lookup` and `br_lookup` as plain Python **dicts** (`merge_key -> TAIL_NUM`),
   which are far lighter than an equivalent DataFrame and give O(1) lookups via
   `.map(dict)`.
2. Read `df_clean_v2.parquet` in batches using `pyarrow.parquet.ParquetFile.iter_batches()`
   (1,000,000 rows per batch) instead of loading it all at once.
3. For each batch: build the US/Brazil merge keys, map them against the lookup dicts to
   attach `TAIL_NUM`, then immediately write that batch to the output file via
   `pyarrow.parquet.ParquetWriter` and discard it from memory (`del` + `gc.collect()`).
4. Match-rate stats are accumulated as running counters across batches rather than
   computed on one giant in-memory frame.

This keeps peak memory to roughly "one batch + the two lookup dicts" regardless of how
large the main dataset grows, rather than requiring the whole dataset to fit in RAM at
once.

---

## Post-Merge Fix: Year-Boundary Leak
After merging, a check of matched years found **12 Brazil rows** with
`scheduled_departure` in 2026 that had nonetheless been matched to a `TAIL_NUM`, even
though only 2022–2025 source files were used to build `br_lookup`. All 12 rows had
`scheduled_departure` of exactly `2026-01-01` — a boundary artifact where the December
2025 Brazil source file contained a small number of individual rows with a `DT_PREVISTO`
that had already rolled into January 1, 2026, even though the file itself is a "2025"
file by name/month. This was not a bug in the file-level year filter, which worked as
intended — it's an artifact of Brazil's per-row scheduled dates not perfectly aligning
with the file's nominal reporting month.

Fix applied: `TAIL_NUM` was nulled out (set to `NaN`) for these 12 specific rows only —
the rows themselves were kept, just stripped of their (out-of-scope) tail number — so
that matched tail numbers are strictly confined to 2022–2025 for both countries.

**Note:** `df1` (the main dataset) itself was never filtered by year — it retains all
years present in `df_clean_v2.parquet`, including 2026 rows. Only the *lookup source
data* was restricted to 2022–2025. Any row outside that range (e.g. 2026 flights)
correctly ends up with `TAIL_NUM = NaN` since there's no lookup data to match against.

---

## Full Code (final version, memory-efficient, streaming)

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
import zipfile
import gc

main_path = r"D:\Departure delay prediction\df_clean_v2.parquet"
output_path = r"D:\Departure delay prediction\df_final.parquet"
needed_cols = ['NR_MOVIMENTO_TIPO', 'NR_AERONAVE_MARCAS', 'NR_AERONAVE_OPERADOR',
               'NR_AEROPORTO_REFERENCIA', 'NR_VOO_OUTRO_AEROPORTO',
               'NR_VOO_NUMERO', 'DT_PREVISTO']

# ======================================================================
# STAGE 1: BUILD BRAZIL LOOKUP (as a dict — cheap to hold in memory)
# ======================================================================

brazil_folder = r"D:\Departure delay prediction\tail_number_brazil"
br_files = sorted([f for f in glob.glob(os.path.join(brazil_folder, "Movimentacoes_Aeroportuarias_*.csv"))
                    if any(str(y) in os.path.basename(f) for y in [2022, 2023, 2024, 2025])])

br_lookup = {}
for fpath in br_files:
    with open(fpath, encoding="latin1") as f:
        first_line = f.readline()
    skip = 0 if "NR_AERONAVE_MARCAS" in first_line else 1

    header = pd.read_csv(fpath, sep=";", encoding="latin1", skiprows=skip, nrows=0, dtype=str)
    col_map = {c: c.replace("ï»¿", "").strip() for c in header.columns}
    use_raw = [raw for raw, clean in col_map.items() if clean in needed_cols]

    df = pd.read_csv(fpath, sep=";", encoding="latin1", skiprows=skip,
                      usecols=use_raw, dtype=str, low_memory=False)
    df = df.rename(columns=col_map)
    df = df[df['NR_MOVIMENTO_TIPO'].isin(['D', 'P'])]
    df['DT_PREVISTO'] = pd.to_datetime(df['DT_PREVISTO'], format='%Y-%m-%d').dt.date
    df['NR_VOO_NUMERO_clean'] = df['NR_VOO_NUMERO'].str.lstrip('0').replace('', '0')

    dep = df[df['NR_MOVIMENTO_TIPO'] == 'D']
    dep_key = (dep['NR_AERONAVE_OPERADOR'] + "_" + dep['NR_VOO_NUMERO_clean'] + "_" +
               dep['NR_AEROPORTO_REFERENCIA'] + "_" + dep['NR_VOO_OUTRO_AEROPORTO'] + "_" +
               dep['DT_PREVISTO'].astype(str))
    for k, v in zip(dep_key, dep['NR_AERONAVE_MARCAS']):
        br_lookup.setdefault(k, v)

    arr = df[df['NR_MOVIMENTO_TIPO'] == 'P']
    arr_key = (arr['NR_AERONAVE_OPERADOR'] + "_" + arr['NR_VOO_NUMERO_clean'] + "_" +
               arr['NR_VOO_OUTRO_AEROPORTO'] + "_" + arr['NR_AEROPORTO_REFERENCIA'] + "_" +
               arr['DT_PREVISTO'].astype(str))
    for k, v in zip(arr_key, arr['NR_AERONAVE_MARCAS']):
        br_lookup.setdefault(k, v)

    del df, dep, arr, dep_key, arr_key

gc.collect()
print("br_lookup entries:", len(br_lookup))

# ======================================================================
# STAGE 2: BUILD US LOOKUP (as a dict)
# ======================================================================

us_folder = r"D:\Departure delay prediction\tail_number_us"
us_files = sorted([f for f in glob.glob(os.path.join(us_folder, "*.zip"))
                    if any(str(y) in f for y in [2022, 2023, 2024, 2025])])

us_lookup = {}
for fpath in us_files:
    with zipfile.ZipFile(fpath) as z:
        inner_name = z.namelist()[0]
        with z.open(inner_name) as f:
            df = pd.read_csv(f, usecols=['FL_DATE', 'OP_UNIQUE_CARRIER', 'TAIL_NUM',
                                          'OP_CARRIER_FL_NUM', 'ORIGIN', 'DEST'], dtype=str)
    df['FL_DATE'] = pd.to_datetime(df['FL_DATE'], format='%m/%d/%Y %I:%M:%S %p').dt.date
    merge_key = (df['OP_UNIQUE_CARRIER'] + "_" + df['OP_CARRIER_FL_NUM'] + "_" +
                 df['ORIGIN'] + "_" + df['DEST'] + "_" + df['FL_DATE'].astype(str))
    for k, v in zip(merge_key, df['TAIL_NUM']):
        us_lookup.setdefault(k, v)
    del df, merge_key

gc.collect()
print("us_lookup entries:", len(us_lookup))

# ======================================================================
# STAGE 3: STREAM df1 IN BATCHES, ATTACH TAIL_NUM, WRITE OUT
# ======================================================================

pf = pq.ParquetFile(main_path)
writer = None
total_rows = 0
matched_us_count = 0
matched_br_count = 0
us_total = 0
br_total = 0

for batch in pf.iter_batches(batch_size=1_000_000):
    chunk = batch.to_pandas()

    chunk['scheduled_departure'] = pd.to_datetime(chunk['scheduled_departure'])
    chunk['flight_date'] = chunk['scheduled_departure'].dt.date

    is_us = chunk['country'] == 'US'
    is_br = chunk['country'] == 'Brazil'

    # ---- US rows ----
    us_key = (
        chunk.loc[is_us, 'airline'].astype(str) + "_" +
        chunk.loc[is_us, 'flight_number'].astype(str) + "_" +
        chunk.loc[is_us, 'origin_airport'].astype(str) + "_" +
        chunk.loc[is_us, 'destination_airport'].astype(str) + "_" +
        chunk.loc[is_us, 'flight_date'].astype(str)
    )

    # ---- Brazil rows ----
    br_flight_num_clean = chunk.loc[is_br, 'flight_number'].astype(str).str.lstrip('0').replace('', '0')
    br_key = (
        chunk.loc[is_br, 'airline'].astype(str) + "_" +
        br_flight_num_clean + "_" +
        chunk.loc[is_br, 'origin_airport'].astype(str) + "_" +
        chunk.loc[is_br, 'destination_airport'].astype(str) + "_" +
        chunk.loc[is_br, 'flight_date'].astype(str)
    )

    chunk['TAIL_NUM'] = pd.NA
    chunk.loc[is_us, 'TAIL_NUM'] = us_key.map(us_lookup)
    chunk.loc[is_br, 'TAIL_NUM'] = br_key.map(br_lookup)

    matched_us_count += chunk.loc[is_us, 'TAIL_NUM'].notna().sum()
    matched_br_count += chunk.loc[is_br, 'TAIL_NUM'].notna().sum()
    us_total += is_us.sum()
    br_total += is_br.sum()

    chunk = chunk.drop(columns=['flight_date'], errors='ignore')

    table = pa.Table.from_pandas(chunk, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(output_path, table.schema)
    writer.write_table(table)

    total_rows += len(chunk)
    del chunk, us_key, br_key, table
    gc.collect()
    print(f"Processed {total_rows} rows so far...")

if writer is not None:
    writer.close()

print(f"US matched: {matched_us_count} / {us_total} = {matched_us_count/us_total*100:.2f}%")
print(f"Brazil matched: {matched_br_count} / {br_total} = {matched_br_count/br_total*100:.2f}%")
print("Saved to", output_path)
```

### Post-processing: null out the 12 year-boundary rows
```python
import pandas as pd

df = pd.read_parquet(r"D:\Departure delay prediction\df_final.parquet")

mask = (df['country'] == 'Brazil') & (df['scheduled_departure'].dt.year == 2026) & (df['TAIL_NUM'].notna())
print("Rows to null out:", mask.sum())

df.loc[mask, 'TAIL_NUM'] = pd.NA

check = df[(df['country'] == 'Brazil') & (df['scheduled_departure'].dt.year == 2026) & (df['TAIL_NUM'].notna())]
print("Remaining 2026 Brazil rows with a tail number (should be 0):", len(check))

df.to_parquet(r"D:\Departure delay prediction\df_final.parquet", index=False)
print("Saved.")
```

---

## Final Result
- **Output file:** `D:\Departure delay prediction\df_final.parquet`
- **Shape:** same row count as `df_clean_v2.parquet` (33,986,255 rows), plus one new
  column, `TAIL_NUM`
- **US match rate:** 90.62% (27,205,150 / 30,021,935)
- **Brazil match rate:** 86.00% (3,409,151 / 3,964,320, after nulling the 12
  year-boundary rows)
- Tail numbers sourced strictly from 2022–2025; 2026 rows in the main dataset (present
  because df1 itself was never year-filtered) correctly carry `TAIL_NUM = NaN`.

## Open Item (not pursued)
Root-causing the remaining ~14% unmatched Brazil rows (flight-number suffix variants
like `950P`, or genuine gaps in the Brazil source data) was identified as a possible next
step but not investigated further — 86% was accepted as a sufficient match rate.
