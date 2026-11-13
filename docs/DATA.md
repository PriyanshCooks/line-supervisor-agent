# Data

The demo reads one shared **events** table (`data/events.db`). Three kinds of source write to it:

| `source` | What it is | How to load it | Committed to git? |
|---|---|---|---|
| `bosch` | Real production data: Kaggle *Bosch Production Line Performance* | `python tasks.py ingest` | No (large, licensed) |
| `vision` | Events from the camera adapter (a 17 s window of a stock clip) | `python tasks.py vision` | No |
| `synthetic` | Bosch-**format** development sample + tiny test fixture | `python tasks.py sample` | Fixture only (`tests/fixtures/`) |

`synthetic` exists so the code, tests and UI run before you have downloaded Kaggle data, and for unit tests.
**It is never shown as real data**: the app prints a warning whenever it is the only source, the agent says so, and every
tool result carries its `source`.

## Download the Bosch data

1. Create a free Kaggle account and open <https://www.kaggle.com/c/bosch-production-line-performance>.
2. On the **Rules** tab, **accept the competition rules** (downloads are blocked until you do).
3. Download `train_date.csv` and `train_numeric.csv` (each is several GB uncompressed; Kaggle serves them zipped).
   - In the browser: *Data* tab, download the two files, unzip.
   - Or with the Kaggle CLI (`pip install kaggle`, API token in `~/.kaggle/kaggle.json`):
     ```bash
     kaggle competitions download -c bosch-production-line-performance -f train_date.csv.zip -p data/bosch
     kaggle competitions download -c bosch-production-line-performance -f train_numeric.csv.zip -p data/bosch
     ```
     then unzip both into `data/bosch/`.
4. You should end up with:
   ```
   data/bosch/train_date.csv
   data/bosch/train_numeric.csv
   ```
   `data/bosch/` is gitignored. Do not commit or redistribute the files (competition rules).

## Ingest

```bash
python tasks.py ingest                       # lines 0 and 3, first 100,000 parts, 8 windows
python -m ingest.bosch --lines 0 1 --max-parts 50000 --windows 12   # options
```

What it does (see `ingest/bosch.py`):
- Reads `train_date.csv` in chunks and only the date columns of the chosen lines (`L{line}_S{station}_D{id}`).
  From `train_numeric.csv` it reads **only** `Id` and `Response`.
- Sample = the **first N parts, in file order, that visit the chosen lines** (a convenience sample, not random).
- Per part and line: entry time at a station = **min** over that station's date columns; route = stations ordered by entry
  time (ties by station number); one `enter` event per visit; one `leave` event per visit that has a next station on the same
  line, timestamped at the **next station's entry**. So `leave - enter` is *time until the part reaches its next recorded
  station* (waiting + processing + transport), **not** a machine cycle time. A part's last station has no `leave`.
- `qc_failed` = `Response` (1 = failed QC), attached to every event of the part.
- Timestamps stay in Bosch's anonymised **time units**. They are never converted to hours.
- The timeline is cut into N equal windows `W1..WN` (by enter time); the tools treat them as shifts.

Measured on the 20,000-part development sample (not real data): ingest ~22 s. I have not run it on the real files yet; expect it to
scale roughly linearly with the number of parts, and the SQLite file to be on the order of 150 MB per 20,000 parts for two lines.

## Dev sample (synthetic)

```bash
python tasks.py sample          # writes data/bosch_sample/ (Bosch-format CSVs) and ingests it with --source synthetic
```

Planted structure so the tests and evals have something to find: slow stations S7 (line 0) and S40 (line 3); S7 slows further
in one time range; S9 (line 0) and S33 (line 3) add QC-failure probability. None of this says anything about real Bosch data.

## Citation

Bosch Production Line Performance, Kaggle competition (2016). Data used under the competition rules; not redistributed here.
