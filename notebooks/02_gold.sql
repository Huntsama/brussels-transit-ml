-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Gold: punctuality per stop and line, and the city hour by hour
-- MAGIC Same logic as brussels-lakehouse `transform/models/gold`.

-- COMMAND ----------

USE CATALOG brussels;
SET TIME ZONE 'UTC';

-- COMMAND ----------

-- Excess Wait Time (EWT), the TfL measure for frequent services:
-- how much longer people really wait than the timetable promises.
-- Actual wait  (AWT): at a random moment, minutes until the next vehicle, averaged over our snapshots.
-- Scheduled wait (SWT): sum(h^2) / (2 * sum(h)) over the timetable headways h in that hour.
CREATE OR REPLACE TABLE gold.stop_line_hourly AS
WITH observed AS (
  SELECT stop_id, line_id, fetched_at, min(minutes_until) AS wait
  FROM silver.stib_waiting
  WHERE minutes_until BETWEEN 0 AND 60
  GROUP BY ALL
),
awt AS (
  SELECT stop_id, line_id, date_trunc('HOUR', fetched_at) AS hour,
         avg(wait) AS actual_wait_min, count(*) AS snapshots
  FROM observed
  GROUP BY ALL
),
headways AS (
  SELECT stop_id, line_id, mode, scheduled_departure,
         (unix_seconds(lead(scheduled_departure) OVER (PARTITION BY stop_id, line_id ORDER BY scheduled_departure))
          - unix_seconds(scheduled_departure)) / 60.0 AS h
  FROM silver.gtfs_departures
),
swt AS (
  SELECT stop_id, line_id, mode, date_trunc('HOUR', scheduled_departure) AS hour,
         sum(h * h) / (2 * sum(h)) AS scheduled_wait_min
  FROM headways
  WHERE h > 0 AND h <= 60
  GROUP BY ALL
)
SELECT stop_id, line_id, mode, hour, snapshots, actual_wait_min, scheduled_wait_min,
       actual_wait_min - scheduled_wait_min AS excess_wait_min
FROM awt
JOIN swt USING (stop_id, line_id, hour);

-- COMMAND ----------

-- One row per hour for Brussels: transit service, air and grid load side by side.
CREATE OR REPLACE TABLE gold.city_hourly AS
WITH transit AS (
  SELECT date_trunc('HOUR', fetched_at) AS hour,
         median(minutes_until) FILTER (WHERE minutes_until BETWEEN 0 AND 60) AS median_wait_min,
         count(DISTINCT stop_id) AS stops_served
  FROM silver.stib_waiting
  GROUP BY 1
),
fleet AS (
  SELECT date_trunc('HOUR', fetched_at) AS hour, count(*) / count(DISTINCT fetched_at) AS vehicles_avg
  FROM silver.stib_vehicles
  GROUP BY 1
),
air AS (
  SELECT date_trunc('HOUR', measured_at) AS hour,
         avg(value_ugm3) FILTER (WHERE pollutant = 'no2') AS no2,
         avg(value_ugm3) FILTER (WHERE pollutant = 'pm10') AS pm10,
         avg(value_ugm3) FILTER (WHERE pollutant = 'pm25') AS pm25
  FROM silver.air_quality
  GROUP BY 1
),
punctuality AS (
  SELECT hour, median(excess_wait_min) AS excess_wait_min FROM gold.stop_line_hourly GROUP BY 1
),
grid AS (
  SELECT date_trunc('HOUR', measured_at) AS hour, avg(load_mw) AS load_mw FROM silver.electricity_load GROUP BY 1
)
SELECT *
FROM transit
FULL JOIN fleet USING (hour)
FULL JOIN air USING (hour)
FULL JOIN punctuality USING (hour)
FULL JOIN grid USING (hour);

-- COMMAND ----------

-- Data checks, the same ones dbt runs. Every count must be 0; the job fails otherwise.
SELECT assert_true(count(*) = 0, 'stop_line_hourly: duplicate stop/line/hour')
FROM (SELECT stop_id, line_id, hour FROM gold.stop_line_hourly GROUP BY ALL HAVING count(*) > 1);

-- COMMAND ----------

SELECT assert_true(count(*) = 0, 'stop_line_hourly: nulls or unknown mode')
FROM gold.stop_line_hourly
WHERE stop_id IS NULL OR line_id IS NULL OR hour IS NULL OR mode NOT IN ('metro', 'tram', 'bus');

-- COMMAND ----------

SELECT assert_true(count(*) = 0, 'air_quality: value outside 0-1000')
FROM silver.air_quality WHERE value_ugm3 NOT BETWEEN 0 AND 1000;

-- COMMAND ----------

SELECT mode, count(*) AS rows, round(median(excess_wait_min), 2) AS median_excess_min
FROM gold.stop_line_hourly GROUP BY mode ORDER BY mode
