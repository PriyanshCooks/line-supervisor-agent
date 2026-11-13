"""Generate a Bosch-FORMAT dev sample (train_date.csv + train_numeric.csv) so everything runs before you download Kaggle data.

THIS IS SYNTHETIC. Ingest it with `--source synthetic` (the default here); never label it 'bosch'.
Planted structure (so tests/evals have something to find):
  * line 0 stations S0..S11, line 3 stations S26..S51; each part skips ~10% of stations
  * S7 (line 0) and S40 (line 3) are slow; S7 slows further in the parts that start in the 55-65% time range
  * S9 (line 0) and S33 (line 3) are visited by ~half the parts and add extra QC-failure probability (+12% / +10%)
    python -m ingest.make_dev_sample --parts 20000 --out data/bosch_sample
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

L0 = list(range(0, 12))
L3 = list(range(26, 52))
BASE_DWELL = {s: 0.25 for s in L0 + L3}
BASE_DWELL.update({7: 1.6, 40: 1.2})
EXTRA_FAIL = {9: 0.12, 33: 0.10}
RARE = {9: 0.50, 33: 0.50}                          # these two stations are visited by only ~half the parts
BASE_FAIL = 0.005


def generate(n_parts: int, out: Path, seed: int = 7, horizon: float = 1700.0) -> dict:
    rng = np.random.default_rng(seed)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.sort(rng.choice(np.arange(1, n_parts * 3), size=n_parts, replace=False))
    start = np.sort(rng.uniform(0, horizon * 0.95, n_parts))
    on0, on3 = rng.random(n_parts) < 0.55, rng.random(n_parts) < 0.70
    on0 |= ~on3                                            # every part is on at least one line
    cols = {}                                              # (line, station) -> list of D column names
    fid = 1
    for ln, sts in ((0, L0), (3, L3)):
        for s in sts:
            k = int(rng.integers(1, 4))
            cols[(ln, s)] = [f"L{ln}_S{s}_D{fid + 2 * j}" for j in range(k)]
            fid += 2 * k
    date = np.full((n_parts, sum(len(v) for v in cols.values())), np.nan)
    colnames = [c for v in cols.values() for c in v]
    cidx = {c: i for i, c in enumerate(colnames)}
    fail_p = np.full(n_parts, BASE_FAIL)
    for i in range(n_parts):
        t = start[i]
        for ln, sts, on in ((0, L0, on0[i]), (3, L3, on3[i])):
            if not on:
                continue
            for s in sts:
                if rng.random() < RARE.get(s, 0.10):       # skip station
                    continue
                for j, c in enumerate(cols[(ln, s)]):
                    date[i, cidx[c]] = round(t + j * 0.01 * rng.random(), 2)
                dwell = BASE_DWELL[s] * rng.lognormal(0, 0.35)
                if s == 7 and 0.55 <= start[i] / horizon <= 0.65:
                    dwell *= 2.2
                t += dwell
                if s in EXTRA_FAIL:
                    fail_p[i] += EXTRA_FAIL[s]
            t += 0.5
    resp = (rng.random(n_parts) < fail_p).astype(int)
    dd = pd.DataFrame(date, columns=colnames)
    dd.insert(0, "Id", ids)
    dd.to_csv(out / "train_date.csv", index=False)
    pd.DataFrame({"Id": ids, "L0_S0_F0": rng.normal(size=n_parts).round(3), "Response": resp}).to_csv(out / "train_numeric.csv", index=False)
    return {"parts": n_parts, "fail_rate": float(resp.mean()), "date_columns": len(colnames), "out": str(out)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=int, default=20000)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "bosch_sample"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(generate(a.parts, Path(a.out), a.seed))


