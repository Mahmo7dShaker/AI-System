# SolarMind — Project README

> Code reference for the files we explained.
> Covered: `main.py` · `dataset_io.py` · `mapping.py` · `models.py` · `preprocess.py`

---

## 1. Project Overview

SolarMind is a web application that takes raw solar energy CSV/Excel files, lets you map columns to the right roles, then preprocesses the data so it is ready for machine learning.

**Data flow:**

1. User uploads a file → `main.py` saves it and stores a row in the DB
2. Dataset page opens → `dataset_io` loads a preview, `mapping.py` guesses columns
3. User confirms mapping → `main.py` saves `mapping_json` to DB
4. User clicks Process → `preprocess.py` cleans the file and writes a CSV

---

## 2. File Structure

```
app/
    main.py          ← API routes  (covered)
    db.py            ← SQLAlchemy engine + Base
    deps.py          ← get_db() dependency
    models.py        ← Dataset, DatasetMapping, ProcessedRun  (covered)
    dataset_io.py    ← load_dataframe()  (covered)
    mapping.py       ← suggest_mapping()  (covered)
ml/features/
    preprocess.py    ← preprocess_with_mapping()  (covered)
templates/           ← index.html, dataset.html
static/css/          ← style.css
datasets/raw/        ← uploaded files
runtime/processed/   ← output CSVs after preprocessing
```

---

## 3. app/main.py

Entry point of the app. FastAPI reads this file when you run uvicorn.

### What it does
- Creates the FastAPI app object
- Calls `Base.metadata.create_all` on startup to create DB tables
- Mounts `/static` and registers the Jinja2 templates folder
- Defines all URL routes (endpoints)

### Endpoints

| Route | What it does |
|---|---|
| `GET /` | Returns `index.html` — the upload form |
| `POST /upload` | Saves the file to `datasets/raw/` and inserts a Dataset row in DB. Redirects to `/dataset/{id}` |
| `GET /datasets` | Returns a JSON list of all uploaded datasets |
| `GET /dataset/{id}` | Renders `dataset.html` with column preview and auto-suggested mapping |
| `POST /dataset/{id}/mapping` | Saves the user-confirmed column mapping as JSON in DatasetMapping table |
| `POST /dataset/{id}/process` | Runs `preprocess_with_mapping()` and writes the clean CSV to `runtime/processed/` |

### Key patterns used
- `uuid4().hex` — generates a unique prefix for every uploaded filename so files never overwrite each other
- `shutil.copyfileobj` — copies the file in chunks, does NOT load the whole file into RAM
- `Depends(get_db)` — FastAPI injects a DB session into every endpoint automatically
- `303 redirect after POST` — browser moves to GET page so refresh does not re-submit the form

### Imports explained

| Import | Why it is needed |
|---|---|
| `FastAPI` | Main framework — creates app, reads routes |
| `UploadFile, File` | Handles multipart file uploads from the HTML form |
| `Form` | Reads form fields sent by POST (mapping dropdowns) |
| `Depends` | Injects shared objects like the DB session |
| `HTMLResponse` | Returns rendered HTML pages |
| `RedirectResponse` | Sends the browser to a new URL after POST |
| `StaticFiles` | Serves `/static` (CSS, images) |
| `Jinja2Templates` | Renders HTML templates with Python variables |
| `Session` | SQLAlchemy type for DB operations |

---

## 4. app/dataset_io.py

Single function: `load_dataframe()`. Opens a file and returns a pandas DataFrame.

### Function signature

```python
def load_dataframe(path: str, nrows: int | None = 200) -> pd.DataFrame:
```

| Parameter | Meaning |
|---|---|
| `path` | Full path to the uploaded file |
| `nrows` | How many rows to read. Default `200` for preview. Pass `None` to read the whole file. |

### How it works
- Converts path to a `Path` object and reads the file extension
- For CSV: tries encodings one by one — `utf-8`, `utf-8-sig`, `cp1256` (Arabic Windows), `latin1`
- If all encodings fail it falls back to pandas default (auto-detect)
- For Excel (`.xlsx` / `.xls`): calls `pd.read_excel()` directly — no encoding loop needed
- Any other extension raises `ValueError` so the caller knows it is unsupported

### Why the encoding loop?

CSV files saved on Arabic Windows use `cp1256`. Files exported from Excel use `utf-8-sig` (adds a BOM byte at the start). By trying the most common encodings in order we avoid crashes on files from different systems.

### Where it is called

| Caller | `nrows` value |
|---|---|
| `GET /dataset/{id}` (preview) | `50` — only need a sample to show columns |
| `POST /dataset/{id}/process` | `None` — need the full file for ML preprocessing |

---

## 5. app/mapping.py

Single function: `suggest_mapping()`. Looks at column names and guesses which column is datetime, which is the energy target, and which are weather features.

### Function signature

```python
def suggest_mapping(columns: list[str]) -> dict:
```

- **Input:** a list of column name strings from the uploaded file
- **Output:** a nested dictionary with keys `datetime`, `plant_id`, `target`, `weather`

### Inner function: find_first

```python
def find_first(patterns):
    for c in cols:
        name = c.lower().strip()
        for pat in patterns:
            if re.search(pat, name):
                return c
    return None
```

- Loops over every column name
- `.lower().strip()` — removes case and whitespace differences before comparing
- `re.search(pat, name)` — checks whether the pattern appears anywhere in the name
- Returns the first column that matches any pattern. Returns `None` if nothing matches.

### Patterns used

| Target column | Regex patterns |
|---|---|
| `datetime` | `date`, `time`, `datetime`, `timestamp` |
| `plant_id` | `plant`, `site`, `station`, `farm`, `id` |
| `target` | `energy`, `kwh`, `power`, `kw`, `output`, `generation` |
| `irradiance` | `irradi`, `ghi`, `dni`, `poa` |
| `ambient_temp` | `ambient`, `air.*temp`, `temp_air` |
| `module_temp` | `module`, `panel.*temp`, `cell.*temp` |
| `wind_speed` | `wind` |
| `cloud_cover` | `cloud` |

### Special regex: `.*`

The pattern `air.*temp` means:
- `air` — literal text
- `.*` — any characters, any count
- `temp` — literal text

Matches: `air_temp`, `air_temperature`, `air_avg_temp` — all with one pattern.

### Example output from your data

```json
{
    "datetime":  "Date/Time",
    "plant_id":  null,
    "target":    "Energy_kWh",
    "weather": {
        "irradiance":   "Irradiation",
        "ambient_temp": "Ambient_Temp",
        "module_temp":  "Module_Temp",
        "wind_speed":   null,
        "cloud_cover":  null
    }
}
```

---

## 6. app/models.py

Defines the 3 database tables using SQLAlchemy ORM. Each Python class becomes one SQL table.

### Dataset table

```python
class Dataset(Base):
    __tablename__ = "datasets"
```

| Column | Purpose |
|---|---|
| `id` | Primary key, auto-increment |
| `original_name` | Filename the user uploaded (e.g. `solar.csv`) |
| `stored_name` | UUID prefix + original name so files never clash |
| `stored_path` | Full path on disk where the file was saved |
| `uploaded_at` | Timestamp — set automatically on insert |

### DatasetMapping table

```python
class DatasetMapping(Base):
    __tablename__ = "dataset_mappings"
```

| Column | Purpose |
|---|---|
| `id` | Primary key |
| `dataset_id` | Foreign key → `datasets.id` |
| `mapping_json` | The full mapping stored as a JSON string (Text column) |
| `created_at` | Timestamp |

> **Why store as JSON text?** Flexible. Different datasets may have different weather columns. No schema migration needed when the mapping structure changes.

### ProcessedRun table

```python
class ProcessedRun(Base):
    __tablename__ = "processed_runs"
```

| Column | Purpose |
|---|---|
| `id` | Primary key |
| `dataset_id` | Foreign key → `datasets.id` |
| `mapping_id` | Foreign key → `dataset_mappings.id` |
| `processed_path` | Where the clean CSV was written |
| `rows_before` | Row count before cleaning |
| `rows_after` | Row count after dropping bad rows |
| `created_at` | Timestamp |

### Common SQLAlchemy column types

| Type | Stores |
|---|---|
| `Integer` | Whole numbers — IDs, row counts |
| `String` | Short text — filenames, paths |
| `Text` | Long text — JSON strings |
| `DateTime` | Date and time values |

---

## 7. ml/features/preprocess.py

Single function: `preprocess_with_mapping()`. Takes raw DataFrame + mapping dict and returns a clean DataFrame ready for ML training.

### Function signature

```python
def preprocess_with_mapping(df: pd.DataFrame, mapping: dict) -> tuple[pd.DataFrame, dict]:
```

Returns two values: the clean DataFrame AND a report dict with `dropped_rows` count.

### Step-by-step pipeline

| Step | What happens |
|---|---|
| 1 — Read mapping | Extract `dt_col`, `target_col`, `plant_col`, `weather` dict from the mapping |
| 2 — Validate | Raise `ValueError` if datetime or target column is missing from the file |
| 3 — Copy DataFrame | `df.copy()` so the original is never modified |
| 4 — Parse datetime | `pd.to_datetime()` converts strings to real datetime objects. Invalid values become `NaT`. |
| 5 — Parse target | `pd.to_numeric()` ensures the energy column is float. Invalid values become `NaN`. |
| 6 — plant_id | Uses the mapped column if it exists, otherwise fills every row with `"default"` |
| 7 — Weather cols | `add_weather()` creates a standard-named column for each weather feature. If the column is not in the file the column is filled with `NaN`. |
| 8 — Drop bad rows | `dropna(subset=["ds_time", "target"])` removes rows where datetime or target is empty |
| 9 — Sort | `sort_values("ds_time")` orders rows chronologically |
| 10 — Time features | Extracts `hour`, `dayofweek`, `month` from `ds_time` for ML feature engineering |
| 11 — Fill weather | `ffill()` fills gaps with the previous value. Remaining `NaN` filled with column median. |

### add_weather helper

```python
def add_weather(new_name, col_name):
    if col_name and col_name in work.columns:
        work[new_name] = pd.to_numeric(work[col_name], errors="coerce")
    else:
        work[new_name] = np.nan
```

Every run produces the same output columns regardless of what the input file contains:

```
irradiance | ambient_temp | module_temp | wind_speed | cloud_cover
```

### Why ffill then median?

- **ffill (forward fill)** — fills a gap with the last known value. Good for time series where the sensor just missed a reading.
- **median fill** — used for gaps at the very start of the file where ffill has no previous value to use. Median is preferred over mean because it is not skewed by extreme outlier values.

### Time features — why extract them?

| Feature | Why it helps the model |
|---|---|
| `hour` | Solar energy peaks at midday. The model needs to know the time of day. |
| `dayofweek` | Consumption patterns differ on weekends vs weekdays. |
| `month` | Summer has more sun than winter. Seasonal pattern. |

### Output columns

```
ds_time | target | plant_id | irradiance | ambient_temp | module_temp | wind_speed | cloud_cover | hour | dayofweek | month
```

---

## 8. Quick Reference

### Run the server

```bash
cd "E:\Spring 2026\A smart"
python -m uvicorn app.main:app --reload
```

### Reset the database

```bash
python scripts\reset_db.py
```

Then restart the server — tables are recreated on startup.

### `errors="coerce"` — what it means

Used in both `pd.to_datetime()` and `pd.to_numeric()`. If a value cannot be converted it becomes `NaN`/`NaT` instead of raising an exception. This lets the pipeline continue and clean bad values later.

### `df.copy()` — why always copy

pandas can silently modify the original DataFrame when you edit a slice. Always working on a copy (`work = df.copy()`) prevents hard-to-find bugs where the original data changes unexpectedly.
