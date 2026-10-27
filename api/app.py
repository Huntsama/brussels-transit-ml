"""Brussels transit status API.

Two published files, combined per line:
  live   line_status.json  last 10-minute window per line, written by notebooks/03_stream_live_status.py
  usual  lines.json        median extra wait per line and hour of day, written by the brussels-lakehouse dashboard

Run: uvicorn app:app --reload   ·   docs at /docs
"""
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from fastapi import FastAPI, HTTPException

BASE = os.environ.get("DATA_BASE_URL", "https://belakehouselake.z1.web.core.windows.net")
CACHE_SECONDS = 60
DISRUPTED_SHARE = 0.2  # the network is "disrupted" when a fifth of the lines are

app = FastAPI(title="Brussels transit status", version="0.1.0")
_cache: dict[str, tuple[float, dict]] = {}


def fetch(name: str) -> dict:
    """GET a published JSON file, cached for a minute so a busy API doesn't hit storage on every request."""
    hit = _cache.get(name)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        return hit[1]
    try:
        r = httpx.get(f"{BASE}/{name}", timeout=10)
        r.raise_for_status()
        data = r.json()
    except (httpx.HTTPError, ValueError):
        if hit:  # storage down: serve the last good copy rather than fail
            return hit[1]
        raise HTTPException(503, f"{name} is not available yet")
    _cache[name] = (time.time(), data)
    return data


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/network/status")
def network_status():
    live = fetch("line_status.json")
    lines = live.get("lines", {})
    disrupted = sorted(k for k, v in lines.items() if v["status"] == "disrupted")
    slower = sorted(k for k, v in lines.items() if v["status"] == "slower")
    share = len(disrupted) / len(lines) if lines else 0
    return {"window_end": live.get("window_end"), "lines": len(lines),
            "status": "disrupted" if share >= DISRUPTED_SHARE else "normal",
            "disrupted": disrupted, "slower": slower}


@app.get("/lines/{line_id}")
def line(line_id: str):
    usual = fetch("lines.json").get("lines", {})
    try:
        live = fetch("line_status.json").get("lines", {})
    except HTTPException:
        live = {}  # usual numbers still work before the stream has run once
    if line_id not in usual and line_id not in live:
        raise HTTPException(404, f"unknown line {line_id}")
    hour = datetime.now(ZoneInfo("Europe/Brussels")).hour
    by_hour = usual.get(line_id, {}).get("by_hour") or [None] * 24
    return {"line_id": line_id, "live": live.get(line_id),
            "usual_extra_min_now": by_hour[hour], "usual_extra_min_by_hour": by_hour}
