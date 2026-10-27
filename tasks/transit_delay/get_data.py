"""Copy the two gold tables out of the lakehouse DuckDB into small Parquet files the study notebook reads.

    uv run python tasks/transit_delay/get_data.py [path/to/lakehouse.duckdb]

The Parquet files are what you upload to Colab (or copy to another laptop): nothing else is needed there.
"""
import os
import sys

import duckdb

DB = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Desktop/Projects/brussels-lakehouse/data/lakehouse.duckdb")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "transit_delay")
os.makedirs(OUT, exist_ok=True)

con = duckdb.connect(DB, read_only=True)
for table in ["stop_line_hourly", "city_hourly"]:
    path = os.path.join(OUT, f"{table}.parquet")
    # hours as UTC so pandas, Colab and Spark all read the same instant
    con.execute(f"""copy (select * replace (hour at time zone 'UTC' as hour) from main_gold.{table} order by hour)
                to '{path}' (format parquet)""")
    n, lo, hi = con.execute(f"select count(*), min(hour)::varchar, max(hour)::varchar from main_gold.{table}").fetchone()
    print(f"{table}: {n:,} rows, {lo} -> {hi}  ->  {path}")
