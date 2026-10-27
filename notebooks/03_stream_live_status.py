# Databricks notebook source
# MAGIC %md
# MAGIC # Live line status: Kafka (Event Hubs) to Delta, with Spark Structured Streaming
# MAGIC
# MAGIC 1. **Read** the `stib-waiting` topic: one event per stop and line every ~20 s (sent by `streaming/producer.py`).
# MAGIC 2. **Bronze**: keep every reading as a Delta table, `bronze.stib_waiting_events`.
# MAGIC 3. **Live status**: per line and 10-minute window, the average wait for the next vehicle, compared with what is
# MAGIC    usual for that line at that hour (from the batch table `gold.stop_line_hourly`).
# MAGIC    More than 2 minutes over usual = `disrupted`: the 12 October strike would have shown up within 10 minutes.
# MAGIC
# MAGIC `trigger` widget: `available_now` processes what's waiting and stops (run it as a job every 15 min, cheap);
# MAGIC `1 minute` keeps running for a live demo (the cluster stays on, so stop it after).

# COMMAND ----------

from pyspark.sql import functions as F

dbutils.widgets.dropdown("trigger", "available_now", ["available_now", "1 minute"])
dbutils.widgets.text("account", "belakehouselake")
trigger = {"availableNow": True} if dbutils.widgets.get("trigger") == "available_now" else {"processingTime": "1 minute"}

spark.sql("USE CATALOG brussels")
spark.sql("CREATE VOLUME IF NOT EXISTS bronze.checkpoints")
CKPT = "/Volumes/brussels/bronze/checkpoints"

# COMMAND ----------

# Event Hubs = Kafka: SASL PLAIN with the connection string as password, kept in a secret scope (infra/streaming.tf)
conn = dbutils.secrets.get("stream", "eventhubs-connection")
namespace = conn.split("sb://")[1].split(".")[0]
kafka = {
    "kafka.bootstrap.servers": f"{namespace}.servicebus.windows.net:9093",
    "kafka.security.protocol": "SASL_SSL",
    "kafka.sasl.mechanism": "PLAIN",
    "kafka.sasl.jaas.config": "kafkashaded.org.apache.kafka.common.security.plain.PlainLoginModule required "
                              f'username="$ConnectionString" password="{conn}";',
    "subscribe": "stib-waiting",
    "startingOffsets": "earliest",
}

schema = ("fetched_at STRING, pointid STRING, lineid STRING, "
          "passingtimes ARRAY<STRUCT<expectedArrivalTime: STRING, lineId: STRING>>")

events = (spark.readStream.format("kafka").options(**kafka).load()
          .select(F.from_json(F.col("value").cast("string"), schema).alias("e")).select("e.*")
          .withColumn("fetched_at", F.to_timestamp("fetched_at"))
          # the next announced arrival at this stop for this line = what someone arriving now would wait
          .withColumn("next", F.expr("filter(passingtimes, t -> t.expectedArrivalTime IS NOT NULL)[0]"))
          .withColumn("minutes_until",
                      (F.unix_timestamp(F.to_timestamp("next.expectedArrivalTime")) - F.unix_timestamp("fetched_at")) / 60)
          .where("minutes_until BETWEEN 0 AND 60")
          .select("fetched_at", F.col("pointid").alias("stop_id"), F.col("lineid").alias("line_id"), "minutes_until"))

# COMMAND ----------

# 2. bronze: every reading, appended
bronze = (events.writeStream.format("delta")
          .option("checkpointLocation", f"{CKPT}/stib_waiting_events")
          .trigger(**trigger)
          .toTable("bronze.stib_waiting_events"))

# COMMAND ----------

# 3. live status per line. "Usual" comes from the batch gold table: a stream-static join.
usual = (spark.table("gold.stop_line_hourly")
         .withColumn("hod", F.hour(F.from_utc_timestamp("hour", "Europe/Brussels")))
         .groupBy("line_id", "hod").agg(F.avg("actual_wait_min").alias("usual_wait_min")))

status = (events
          .withWatermark("fetched_at", "10 minutes")  # readings more than 10 min late are dropped
          .groupBy(F.window("fetched_at", "10 minutes"), "line_id")
          .agg(F.avg("minutes_until").alias("wait_min"), F.approx_count_distinct("stop_id").alias("stops"))
          .withColumn("hod", F.hour(F.from_utc_timestamp(F.col("window.end"), "Europe/Brussels")))
          .join(usual, ["line_id", "hod"], "left")
          .withColumn("extra_min", F.col("wait_min") - F.col("usual_wait_min"))
          .withColumn("status", F.when(F.col("extra_min") > 2, "disrupted")
                                 .when(F.col("extra_min") > 1, "slower").otherwise("normal"))
          .select(F.col("window.start").alias("window_start"), F.col("window.end").alias("window_end"),
                  "line_id", "wait_min", "usual_wait_min", "extra_min", "stops", "status"))

live = (status.writeStream.format("delta").outputMode("append")
        .option("checkpointLocation", f"{CKPT}/line_status_live")
        .trigger(**trigger)
        .toTable("gold.line_status_live"))

# COMMAND ----------

# with available_now both queries stop by themselves once the backlog is processed
for q in (bronze, live):
    q.awaitTermination()

latest = spark.sql("""
  SELECT * FROM gold.line_status_live
  WHERE window_end = (SELECT max(window_end) FROM gold.line_status_live)
  ORDER BY extra_min DESC""")
display(latest)

# COMMAND ----------

# Publish the latest window as a small JSON file next to the dashboard: the API (api/app.py) and the website read it.
import json

rows = latest.collect()
if rows:
    payload = {"window_end": rows[0].window_end.isoformat() + "Z", "lines": {
        r.line_id: {"status": r.status, "wait_min": round(r.wait_min, 2), "stops": r.stops,
                    "usual_wait_min": None if r.usual_wait_min is None else round(r.usual_wait_min, 2),
                    "extra_min": None if r.extra_min is None else round(r.extra_min, 2)} for r in rows}}
    dbutils.fs.put(f"abfss://$web@{dbutils.widgets.get('account')}.dfs.core.windows.net/line_status.json",
                   json.dumps(payload), overwrite=True)
