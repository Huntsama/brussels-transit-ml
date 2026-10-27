"""Kafka producer: STIB's live feed into Azure Event Hubs, one event per stop and line, every 20 seconds.

The batch lakehouse only keeps one STIB snapshot every 5 minutes. STIB refreshes every ~20 s, so this sends every
reading as it comes, which is what the live disruption check in Databricks needs.

Topics (Event Hubs "event hubs" = Kafka topics):
  stib-waiting    one message per (stop, line): the next arrivals announced there
  stib-vehicles   one message per line: where its vehicles are

Env: KAFKA_BOOTSTRAP  <namespace>.servicebus.windows.net:9093
     KAFKA_CONNECTION Event Hubs connection string (Terraform output `eventhubs_send_connection`)
     POLL_SECONDS     default 20
"""
import json
import os
import time
import urllib.request
from datetime import datetime, timezone

from confluent_kafka import Producer

STIB = "https://api-management-discovery-production.azure-api.net/api/datasets/stibmivb/rt"
POLL = int(os.environ.get("POLL_SECONDS", "20"))


def get(path):
    req = urllib.request.Request(f"{STIB}/{path}", headers={"User-Agent": "brussels-transit-ml"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["results"]


def messages(waiting, vehicles, fetched_at):
    """Flatten STIB's nested JSON-in-JSON into (topic, key, value) triples; the key keeps a stop's events in order."""
    for r in waiting:
        yield "stib-waiting", f"{r['pointid']}|{r['lineid']}", {
            "fetched_at": fetched_at, "pointid": r["pointid"], "lineid": r["lineid"],
            "passingtimes": json.loads(r["passingtimes"])}
    for r in vehicles:
        yield "stib-vehicles", r["lineid"], {
            "fetched_at": fetched_at, "lineid": r["lineid"], "positions": json.loads(r["vehiclepositions"])}


def producer():
    # Event Hubs speaks the Kafka protocol: SASL PLAIN, user "$ConnectionString", password = the connection string
    return Producer({
        "bootstrap.servers": os.environ["KAFKA_BOOTSTRAP"],
        "security.protocol": "SASL_SSL", "sasl.mechanisms": "PLAIN",
        "sasl.username": "$ConnectionString", "sasl.password": os.environ["KAFKA_CONNECTION"],
        "linger.ms": 100, "compression.type": "gzip",
    })


def main():
    p = producer()
    while True:
        started = time.time()
        try:
            now = datetime.now(timezone.utc).isoformat()
            n = 0
            for topic, key, value in messages(get("WaitingTimes"), get("VehiclePositions"), now):
                p.produce(topic, key=key, value=json.dumps(value))
                n += 1
            p.flush(30)
            print(f"{now} sent {n} events", flush=True)
        except Exception as e:  # a failed poll must not kill the stream; the next one retries
            print(f"poll failed: {e!r}", flush=True)
        time.sleep(max(0, POLL - (time.time() - started)))


if __name__ == "__main__":
    main()
