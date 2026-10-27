"""One-shot local demo of what the streaming pipeline produces, without Kafka or Databricks.

Same logic as notebooks/03_stream_live_status.py, on a single STIB reading instead of a 10-minute stream:
next-vehicle wait per line right now, compared with the usual wait for that line at this hour (from the
collected history in data/transit_delay/stop_line_hourly.parquet). Writes demo/out/line_status.json.

    uv run python demo/live_demo.py
"""
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "demo" / "out"
STIB = "https://api-management-discovery-production.azure-api.net/api/datasets/stibmivb/rt/WaitingTimes"

sys.path.insert(0, str(ROOT / "streaming"))
sys.modules.setdefault("confluent_kafka", type(sys)("confluent_kafka"))
sys.modules["confluent_kafka"].Producer = None
from producer import messages  # noqa: E402  the producer's own flattening, so the demo matches the stream

req = urllib.request.Request(STIB, headers={"User-Agent": "brussels-transit-ml"})
now = datetime.now(timezone.utc)
rows = json.load(urllib.request.urlopen(req, timeout=30))["results"]
events = [v for t, _, v in messages(rows, [], now.isoformat()) if t == "stib-waiting"]
print(f"STIB right now: {len(events):,} events (one per stop and line), as the producer would send them to Kafka")

waits = []
for e in events:
    nxt = [p for p in e["passingtimes"] if p.get("expectedArrivalTime")]
    if nxt:
        m = (datetime.fromisoformat(nxt[0]["expectedArrivalTime"]) - now).total_seconds() / 60
        if 0 <= m <= 60:
            waits.append((e["lineid"], e["pointid"], m))
live = pd.DataFrame(waits, columns=["line_id", "stop_id", "minutes_until"])
per_line = live.groupby("line_id").agg(wait_min=("minutes_until", "mean"), stops=("stop_id", "nunique"))

hist = pd.read_parquet(ROOT / "data" / "transit_delay" / "stop_line_hourly.parquet",
                       columns=["line_id", "hour", "actual_wait_min"])
hod = now.astimezone(tz=pd.Timestamp.now(tz="Europe/Brussels").tz).hour
hist["hod"] = pd.to_datetime(hist["hour"], utc=True).dt.tz_convert("Europe/Brussels").dt.hour
usual = hist[hist["hod"] == hod].groupby("line_id")["actual_wait_min"].mean().rename("usual_wait_min")

s = per_line.join(usual, how="left")
s["extra_min"] = s["wait_min"] - s["usual_wait_min"]
s["status"] = pd.cut(s["extra_min"], [-999, 1, 2, 999], labels=["normal", "slower", "disrupted"]).astype(str)
s.loc[s["usual_wait_min"].isna(), "status"] = "normal"  # no history for this line at this hour: don't guess

OUT.mkdir(parents=True, exist_ok=True)
payload = {"window_end": now.isoformat(), "lines": {
    k: {"status": r.status, "wait_min": round(r.wait_min, 2), "stops": int(r.stops),
        "usual_wait_min": None if pd.isna(r.usual_wait_min) else round(r.usual_wait_min, 2),
        "extra_min": None if pd.isna(r.extra_min) else round(r.extra_min, 2)} for k, r in s.iterrows()}}
(OUT / "line_status.json").write_text(json.dumps(payload, indent=1))

print(f"\n{len(s)} lines at {now.astimezone():%H:%M} (Brussels hour {hod}:00)")
print(s["status"].value_counts().to_string())
print("\nFurthest above usual:")
print(s.sort_values("extra_min", ascending=False).head(8).round(1).to_string())
print(f"\nwrote {OUT / 'line_status.json'}")
