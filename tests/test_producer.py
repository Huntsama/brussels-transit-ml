import json
import sys
import types
from pathlib import Path

# the test only needs the flattening, not a Kafka client
sys.modules.setdefault("confluent_kafka", types.SimpleNamespace(Producer=None))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "streaming"))
from producer import messages  # noqa: E402


def test_messages_flatten_stib_json_strings():
    waiting = [{"pointid": "8042", "lineid": "71",
                "passingtimes": json.dumps([{"expectedArrivalTime": "2026-10-28T08:03:00+01:00"}])}]
    vehicles = [{"lineid": "71", "vehiclepositions": json.dumps([{"pointId": "8042", "distanceFromPoint": 120}])}]
    out = list(messages(waiting, vehicles, "2026-10-28T07:00:00+00:00"))
    assert [(t, k) for t, k, _ in out] == [("stib-waiting", "8042|71"), ("stib-vehicles", "71")]
    assert out[0][2]["passingtimes"][0]["expectedArrivalTime"].startswith("2026-10-28")
    assert out[1][2]["positions"][0]["distanceFromPoint"] == 120
