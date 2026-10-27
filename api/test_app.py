import pytest
from fastapi.testclient import TestClient

import app as api

LIVE = {"window_end": "2026-10-28T08:10:00Z", "lines": {
    "71": {"status": "disrupted", "extra_min": 3.1},
    "1": {"status": "normal", "extra_min": 0.1},
    "46": {"status": "slower", "extra_min": 1.4},
    "92": {"status": "normal", "extra_min": 0.2},
}}
USUAL = {"lines": {"71": {"by_hour": [0.5] * 24, "excess": 0.5}, "7": {"by_hour": [0.2] * 24, "excess": 0.2}}}


@pytest.fixture
def client(monkeypatch):
    files = {"line_status.json": LIVE, "lines.json": USUAL}
    monkeypatch.setattr(api, "fetch", lambda name: files[name])
    return TestClient(api.app)


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_network_status_flags_a_disrupted_network(client):
    body = client.get("/network/status").json()
    assert body["status"] == "disrupted"  # 1 of 4 lines = 25%, over the 20% threshold
    assert body["disrupted"] == ["71"] and body["slower"] == ["46"]


def test_line_combines_live_and_usual(client):
    body = client.get("/lines/71").json()
    assert body["live"]["status"] == "disrupted"
    assert body["usual_extra_min_now"] == 0.5


def test_line_without_live_data_still_answers(client):
    assert client.get("/lines/7").json()["live"] is None


def test_unknown_line_is_404(client):
    assert client.get("/lines/999").status_code == 404


def test_storage_down_serves_last_good_copy(monkeypatch):
    api._cache["x.json"] = (0, {"cached": True})  # expired entry

    def boom(*a, **k):
        raise api.httpx.ConnectError("down")
    monkeypatch.setattr(api.httpx, "get", boom)
    assert api.fetch("x.json") == {"cached": True}
