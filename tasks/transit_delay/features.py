"""Step 1: build the model table with DuckDB, straight from the Parquet files (no full pandas load).

    uv run python tasks/transit_delay/features.py [--data DIR]

Writes, next to the inputs:
  features.parquet       one row per stop x line x hour on full days, compact types (float32, small ints, dictionary strings)
  per_day.csv            rows, hours, median excess wait, full/disrupted flag per local day (for the report)
  features.meta.json     data snapshot id (sha256 of the inputs + first/last hour), row counts, disrupted days
The SQL is in features.sql; the rules are explained in the notebook.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SQL = (HERE / "features.sql").read_text()

# compact types for the Parquet file; the SQL itself computes in double precision
COMPACT = {
    "stop_id": None, "line_id": None, "mode": None, "hour": None, "date": None,
    "hod": "TINYINT", "dow": "TINYINT", "disrupted": None,
    "scheduled_wait_min": "FLOAT", "excess_wait_min": "FLOAT", "prev_hour_ewt": "FLOAT",
    "prev_days_same_hod_ewt": "FLOAT", "city_prev_ewt": "FLOAT", "no2_prev": "FLOAT", "load_prev": "FLOAT",
}


def connect(memory_limit="2GB", threads=4):
    con = duckdb.connect()
    con.execute(f"set memory_limit='{memory_limit}'; set threads={threads}; set preserve_insertion_order=false")
    return con


def build(con, slh_path, city_path):
    """Run features.sql on the two Parquet files; leaves temp tables `per_day` and `features` (double precision)."""
    con.execute(f"create or replace temp view slh_raw as select * replace (hour::timestamp as hour) from read_parquet('{slh_path}')")
    con.execute(f"create or replace temp view city_raw as select * replace (hour::timestamp as hour) from read_parquet('{city_path}')")
    con.execute(SQL)
    return con


def compact_select():
    return ", ".join(f"{c}::{t} as {c}" if t else c for c, t in COMPACT.items())


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def snapshot_id(con, slh_path, city_path):
    lo, hi, n = con.execute("select min(hour)::varchar, max(hour)::varchar, count(*) from slh_raw").fetchone()
    inputs = {Path(p).name: sha256(p) for p in (slh_path, city_path)}
    combined = hashlib.sha256("".join(inputs.values()).encode()).hexdigest()[:16]
    return {"snapshot_id": f"{combined}_{lo[:13].replace(' ', 'T')}_{hi[:13].replace(' ', 'T')}",
            "inputs_sha256": inputs, "first_hour_utc": lo, "last_hour_utc": hi, "input_rows": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "transit_delay"))
    args = ap.parse_args()
    data = Path(args.data)
    slh, city = data / "stop_line_hourly.parquet", data / "city_hourly.parquet"
    con = build(connect(), slh, city)
    out = data / "features.parquet"
    con.execute(f"copy (select {compact_select()} from features order by hour, stop_id, line_id) "
                f"to '{out}' (format parquet, compression zstd)")
    con.execute(f"copy (select * from per_day order by date) to '{data / 'per_day.csv'}' (header)")
    meta = snapshot_id(con, slh, city)
    meta["feature_rows"] = con.execute("select count(*) from features").fetchone()[0]
    meta["disrupted_days"] = [str(d) for (d,) in con.execute("select date from per_day where disrupted").fetchall()]
    meta["partial_days"] = [str(d) for (d,) in con.execute("select date from per_day where not full_day").fetchall()]
    (data / "features.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{meta['feature_rows']:,} rows -> {out} ({out.stat().st_size / 1e6:.0f} MB)")
    print(json.dumps({k: meta[k] for k in ["snapshot_id", "disrupted_days", "partial_days"]}, indent=2))


if __name__ == "__main__":
    main()
