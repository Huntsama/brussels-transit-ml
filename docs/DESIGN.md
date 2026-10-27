# System design: Brussels transit, batch + streaming

## The problem
STIB publishes where its vehicles are and when the next ones arrive, refreshed about every 20 seconds, and keeps no
history. The batch lakehouse saves one snapshot every 5 minutes: enough to study delays afterwards, too slow to
say "line 71 is disrupted *right now*". On 12 October (national strike) the whole network ran 4+ minutes late all
day; a rider would have wanted to know that at 7 a.m., not the next morning.

## The design

```
                         ┌──────────── batch (exists) ────────────┐
STIB API ─ every 5 min ─►│ Azure Function ─► ADLS bronze (JSON)   │
                         │   dbt/DuckDB or Databricks ─► silver ─► gold: stop_line_hourly ("usual")
                         └────────────────────────────────────────┘
                                                                         │ usual wait per line and hour
STIB API ─ every 20 s ─► producer (Container App) ─► Kafka = Event Hubs ──┤
                         topics: stib-waiting, stib-vehicles              ▼
                         Databricks Structured Streaming, every 15 min (or continuous for a demo)
                           ├─► bronze.stib_waiting_events   (every reading, Delta)
                           └─► gold.line_status_live        (10-min windows: wait vs usual → normal / slower / disrupted)
                                     │
                                     ├─► line_status.json (static site) ─► FastAPI on Container Apps ─► website, apps
                                     └─► Power BI report (Databricks connector)
```

## Why Kafka here
Kafka is a **buffer between a source that never stops and consumers that don't run all the time**.
- The producer only reads STIB and writes events; it knows nothing about Spark, tables or the API.
- Databricks can be off (cheap) and catch up later: Kafka keeps the events (1 day here) and Spark remembers how far
  it read (the checkpoint), so nothing is lost or read twice.
- Several consumers can read the same topic independently: today Spark, tomorrow an alerting service.
- Event Hubs speaks the Kafka protocol, so the code is plain Kafka (`confluent-kafka`, Spark's `kafka` source) and
  would work unchanged against Confluent or a self-hosted cluster.

What is streamed: every STIB reading, one event per stop and line (`stib-waiting`) and per line (`stib-vehicles`),
about 2,000 events every 20 seconds. The key (`stop|line`) sends all events of one stop to the same partition, so
they stay in order.

## Decisions and trade-offs
| Decision | Why | Cost of it |
|---|---|---|
| Spark job every 15 min with `availableNow`, not an always-on stream | A cluster running 24/7 would eat the student credit in days | Status is up to 15 min old; switch the widget to `1 minute` for a live demo |
| 10-minute windows, 10-minute watermark | Smooths single late vehicles; late readings beyond 10 min are dropped | One bad bus can't flag a line, but a sudden failure shows up only after the window closes |
| "Disrupted" = > 2 min over usual for that line and hour | Usual comes from the batch gold table, so rush hour isn't mistaken for a disruption | A fixed threshold; a model (the `transit_delay` task) could replace it |
| API reads a published JSON, not the Delta table | No database or SQL warehouse to pay for; the API scales to zero | The API is read-only and as fresh as the last Spark run |
| API caches each file for 60 s and serves the last good copy if storage fails | Fewer storage reads; degraded but working when storage is down | Data up to a minute older than the file |

## Failure modes
- **STIB API down:** the producer logs the error and tries again 20 s later; Kafka simply gets no events.
- **Producer down:** Container Apps restarts it; no events until it's back; the batch lakehouse still has its 5-min snapshots.
- **Databricks job fails:** e-mail; Kafka keeps a day of events, the next run catches up from the checkpoint.
- **Storage down:** the API serves its last cached copy, or 503 with a clear message.

## Scaling
At 2,000 events / 20 s (~100/s) one partition per topic would do; 2 partitions allow 2 parallel Spark readers.
For all of Belgium (De Lijn, TEC, SNCB) add producers and topics; Event Hubs Standard handles 1 MB/s per throughput unit.

## Power BI
Power BI Desktop is Windows-only. Two ways from Linux:
1. In Databricks: Catalog → `gold.line_status_live` → **Use with BI tools → Power BI**: downloads a connection file;
   open it in Power BI on a Windows machine (VUB computer, a friend's laptop, or an Azure Windows VM for an hour).
2. Power BI service (app.powerbi.com) with your VUB account: Get data → Azure Databricks.
Report: map of stops coloured by extra wait, line status over the day, "usual vs today" per line, the 12 October
strike as the headline example.

## Cost (student credit)
| Piece | While running |
|---|---|
| Event Hubs Standard (Kafka) | ~0.03 €/h ≈ 22 €/month |
| Producer container (0.25 vCPU) | ≈ 5 €/month |
| API container (scales to zero) | ≈ 0 € |
| Databricks job, 15-min runs from 6:00 to 24:00 on one DS3_v2 | ~0.30 €/h of VM while running + DBUs after the 14-day trial |

Keep it on for a demo period, then remove the streaming part:
`terraform destroy -target=azurerm_eventhub_namespace.stream -target=azurerm_container_app.producer`

## Your steps
1. Push to `main` → GitHub Actions runs the tests and publishes `ghcr.io/huntsama/transit-api` and
   `ghcr.io/huntsama/stib-producer`. Once, on GitHub: Packages → each package → Package settings → visibility **Public**
   (so Azure can pull them without a password).
2. `cd infra && terraform apply -target=azurerm_databricks_workspace.dbw -target=azurerm_role_assignment.connector_writes_lake`,
   then `terraform apply`.
3. Run the batch job once (`job_url` in the outputs) so `gold.stop_line_hourly` exists.
4. In Databricks, open `/Shared/brussels/03_stream_live_status`, run it with `trigger = 1 minute` for a live look,
   then unpause the `stream-live-status` job for the 15-minute schedule.
5. Open `api_url` from the Terraform outputs: `/docs`, `/network/status`, `/lines/71`.
