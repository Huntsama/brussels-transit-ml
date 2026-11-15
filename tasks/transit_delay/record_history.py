"""Append this run to outputs/transit_delay/history.csv and check the model still earns its place.

    uv run python tasks/transit_delay/record_history.py [--min-gain 3]

One row per run: the data snapshot, the standard-split MAE of LightGBM and of the best simple rule, and the
backtest mean. Exit code 1 when LightGBM is less than --min-gain % better than the best rule on the newest
backtest week: the weekly retrain then fails and GitHub emails the owner. A model that no longer beats a rule
should not be presented as the answer.
"""
import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "transit_delay"
MODEL = "LightGBM without air/grid/city features"
RULES = ["Average per stop/line/hour of day", "Always 0 (trust the timetable)", "Same as one hour ago"]


def summarise(metrics: pd.Series, backtest: pd.DataFrame, meta: dict) -> dict:
    best_rule = metrics[RULES].min()
    newest = backtest[backtest.fold == backtest.fold.min()].set_index("model")["mae"]
    week_rule = newest[RULES].min()
    lgbm_bt = backtest[backtest.model == MODEL]["mae"]
    return {
        "run_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        "snapshot_id": meta["snapshot_id"],
        "data_from": meta["first_hour_utc"][:10], "data_to": meta["last_hour_utc"][:10],
        "rows": meta["input_rows"],
        "lgbm_mae": round(float(metrics[MODEL]), 4), "best_rule_mae": round(float(best_rule), 4),
        "gain_pct": round(100 * (1 - metrics[MODEL] / best_rule), 1),
        "backtest_folds": int(backtest.fold.nunique()),
        "backtest_lgbm_mean": round(float(lgbm_bt.mean()), 4),
        "backtest_lgbm_std": round(float(lgbm_bt.std(ddof=0)), 4),
        "newest_week_gain_pct": round(100 * (1 - newest[MODEL] / week_rule), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-gain", type=float, default=3.0, help="percent LightGBM must beat the best rule by")
    args = ap.parse_args()
    metrics = pd.read_csv(OUT / "metrics.csv", index_col=0)["mae_min"]
    backtest = pd.read_csv(OUT / "backtest.csv")
    meta = json.loads((ROOT / "data" / "transit_delay" / "features.meta.json").read_text())
    row = summarise(metrics, backtest, meta)

    path = OUT / "history.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(json.dumps(row, indent=1))
    if row["newest_week_gain_pct"] < args.min_gain:
        print(f"::error::LightGBM is only {row['newest_week_gain_pct']}% better than the best rule on the newest week "
              f"(minimum {args.min_gain}%)")
        sys.exit(1)


if __name__ == "__main__":
    main()
