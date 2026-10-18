-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Silver: bronze JSON to clean Delta tables
-- MAGIC Same logic as the dbt models in brussels-lakehouse (`transform/models/silver`), in Spark SQL.
-- MAGIC Each table is rebuilt from all of bronze: ~500 MB, a few minutes on one node.
-- MAGIC <!-- ponytail: full rebuild; switch to Auto Loader (cloudFiles, availableNow) once bronze outgrows a single run -->

-- COMMAND ----------

USE CATALOG brussels;
SET TIME ZONE 'UTC';

-- COMMAND ----------

-- One row per (snapshot, stop, line, upcoming arrival).
CREATE OR REPLACE TABLE silver.stib_waiting AS
WITH raw AS (
  SELECT cast(fetched_at AS timestamp) AS fetched_at, explode(data.results) AS r
  FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/stib_waiting/*/*', format => 'json')
)
SELECT
  fetched_at,
  r.pointid AS stop_id,
  r.lineid AS line_id,
  cast(p.expectedArrivalTime AS timestamp) AS expected_arrival,
  (unix_millis(cast(p.expectedArrivalTime AS timestamp)) - unix_millis(fetched_at)) / 60000.0 AS minutes_until
FROM raw
LATERAL VIEW explode(from_json(r.passingtimes, 'array<struct<expectedArrivalTime:string>>')) AS p
WHERE p.expectedArrivalTime IS NOT NULL;

-- COMMAND ----------

-- One row per vehicle seen in a snapshot.
CREATE OR REPLACE TABLE silver.stib_vehicles AS
WITH raw AS (
  SELECT cast(fetched_at AS timestamp) AS fetched_at, explode(data.results) AS r
  FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/stib_vehicles/*/*', format => 'json')
)
SELECT fetched_at, r.lineid AS line_id, p.directionId AS direction_id, p.pointId AS stop_id,
       p.distanceFromPoint AS metres_from_stop
FROM raw
LATERAL VIEW explode(from_json(r.vehiclepositions,
  'array<struct<directionId:string, pointId:string, distanceFromPoint:int>>')) AS p;

-- COMMAND ----------

-- Latest known name and position per stop.
CREATE OR REPLACE TABLE silver.stib_stops AS
WITH raw AS (
  SELECT cast(fetched_at AS timestamp) AS fetched_at, explode(data.results) AS r
  FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/stib_stops/*/*', format => 'json')
)
SELECT
  r.id AS stop_id,
  from_json(r.name, 'fr string, nl string').fr AS name_fr,
  from_json(r.name, 'fr string, nl string').nl AS name_nl,
  from_json(r.gpscoordinates, 'latitude double, longitude double').latitude AS lat,
  from_json(r.gpscoordinates, 'latitude double, longitude double').longitude AS lon
FROM raw
QUALIFY row_number() OVER (PARTITION BY r.id ORDER BY fetched_at DESC) = 1;

-- COMMAND ----------

-- Hourly pollutant values per Brussels station; snapshots overlap, keep one row per hour.
CREATE OR REPLACE TABLE silver.air_quality AS
WITH raw AS (
  SELECT cast(fetched_at AS timestamp) AS fetched_at, explode(data) AS s
  FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/ircel/*/*', format => 'json')
)
SELECT
  s.station, s.pollutant, s.lat, s.lon,
  timestamp_millis(x.timestamp) AS measured_at,
  greatest(x.value, 0) AS value_ugm3  -- PM sensors read slightly below 0 on clean nights
FROM raw
LATERAL VIEW explode(s.`values`) AS x
WHERE x.value IS NOT NULL
QUALIFY row_number() OVER (PARTITION BY s.station, s.pollutant, x.timestamp ORDER BY fetched_at DESC) = 1;

-- COMMAND ----------

-- Belgian grid load (MW), 15-minute resolution; latest revision wins.
CREATE OR REPLACE TABLE silver.electricity_load AS
WITH raw AS (
  SELECT cast(fetched_at AS timestamp) AS fetched_at, explode(data.results) AS r
  FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/elia/*/*', format => 'json')
)
SELECT cast(r.datetime AS timestamp) AS measured_at, r.totalload AS load_mw
FROM raw
QUALIFY row_number() OVER (PARTITION BY r.datetime ORDER BY fetched_at DESC) = 1;

-- COMMAND ----------

-- MAGIC %md
-- MAGIC ## Timetable (GTFS)
-- MAGIC Only the newest weekly download is used. `partitionColumns => ''` stops the `date=` folder from clashing with
-- MAGIC GTFS's own `date` column in `calendar_dates.txt`.

-- COMMAND ----------

CREATE OR REPLACE TEMPORARY VIEW gtfs_latest AS
SELECT max(regexp_extract(_metadata.file_path, 'date=([0-9-]+)', 1)) AS d
FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/gtfs/*/routes.txt',
                format => 'csv', header => true, inferSchema => false, partitionColumns => '');

-- COMMAND ----------

-- One row per line: STIB's own colours, for dashboard badges.
CREATE OR REPLACE TABLE silver.gtfs_routes AS
SELECT route_short_name AS line_id,
       '#' || route_color AS color,
       '#' || coalesce(nullif(route_text_color, ''), 'FFFFFF') AS text_color
FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/gtfs/*/routes.txt',
                format => 'csv', header => true, inferSchema => false, partitionColumns => '')
WHERE _metadata.file_path LIKE '%date=' || (SELECT d FROM gtfs_latest) || '%'
QUALIFY row_number() OVER (PARTITION BY route_short_name ORDER BY route_id) = 1;

-- COMMAND ----------

-- Scheduled departures per stop and line, for the days we have live snapshots of.
CREATE OR REPLACE TABLE silver.gtfs_departures AS
WITH
file AS (SELECT '%date=' || d || '%' AS pattern FROM gtfs_latest),
routes AS (
  SELECT * FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/gtfs/*/routes.txt',
    format => 'csv', header => true, inferSchema => false, partitionColumns => '')
  WHERE _metadata.file_path LIKE (SELECT pattern FROM file)
),
trips AS (
  SELECT * FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/gtfs/*/trips.txt',
    format => 'csv', header => true, inferSchema => false, partitionColumns => '')
  WHERE _metadata.file_path LIKE (SELECT pattern FROM file)
),
stop_times AS (
  SELECT trip_id, stop_id, departure_time
  FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/gtfs/*/stop_times.txt',
    format => 'csv', header => true, inferSchema => false, partitionColumns => '')
  WHERE _metadata.file_path LIKE (SELECT pattern FROM file)
),
calendar AS (
  SELECT * FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/gtfs/*/calendar.txt',
    format => 'csv', header => true, inferSchema => false, partitionColumns => '')
  WHERE _metadata.file_path LIKE (SELECT pattern FROM file)
),
calendar_dates AS (
  SELECT service_id, to_date(`date`, 'yyyyMMdd') AS d, exception_type
  FROM read_files('abfss://bronze@belakehouselake.dfs.core.windows.net/gtfs/*/calendar_dates.txt',
    format => 'csv', header => true, inferSchema => false, partitionColumns => '')
  WHERE _metadata.file_path LIKE (SELECT pattern FROM file)
),
-- Trips after midnight belong to the previous service day, so include the day before too.
days AS (
  SELECT DISTINCT d FROM (
    SELECT to_date(from_utc_timestamp(fetched_at, 'Europe/Brussels')) AS d FROM silver.stib_waiting
    UNION ALL
    SELECT date_sub(to_date(from_utc_timestamp(fetched_at, 'Europe/Brussels')), 1) FROM silver.stib_waiting
  )
),
active AS (
  (
    SELECT c.service_id, days.d
    FROM calendar c
    JOIN days ON days.d BETWEEN to_date(c.start_date, 'yyyyMMdd') AND to_date(c.end_date, 'yyyyMMdd')
    -- dayofweek: 1 = Sunday
    WHERE element_at(array(c.sunday, c.monday, c.tuesday, c.wednesday, c.thursday, c.friday, c.saturday),
                     dayofweek(days.d)) = '1'
    UNION
    SELECT service_id, d FROM calendar_dates WHERE exception_type = '1' AND d IN (SELECT d FROM days)
  )
  EXCEPT
  SELECT service_id, d FROM calendar_dates WHERE exception_type = '2'
)
SELECT
  st.stop_id,
  r.route_short_name AS line_id,
  -- GTFS route_type: 0 tram, 1 metro, 3 bus
  CASE r.route_type WHEN '0' THEN 'tram' WHEN '1' THEN 'metro' WHEN '3' THEN 'bus' ELSE 'other' END AS mode,
  -- GTFS times can pass 24:00: seconds from local midnight of the service day, then local time to UTC.
  to_utc_timestamp(timestamp_seconds(unix_seconds(cast(a.d AS timestamp))
    + int(split_part(st.departure_time, ':', 1)) * 3600
    + int(split_part(st.departure_time, ':', 2)) * 60
    + int(split_part(st.departure_time, ':', 3))), 'Europe/Brussels') AS scheduled_departure
FROM stop_times st
JOIN trips t USING (trip_id)
JOIN routes r ON r.route_id = t.route_id
JOIN active a ON a.service_id = t.service_id;

-- COMMAND ----------

-- Row counts, to compare with the DuckDB build.
SELECT 'stib_waiting' AS t, count(*) AS n FROM silver.stib_waiting UNION ALL
SELECT 'stib_vehicles', count(*) FROM silver.stib_vehicles UNION ALL
SELECT 'stib_stops', count(*) FROM silver.stib_stops UNION ALL
SELECT 'air_quality', count(*) FROM silver.air_quality UNION ALL
SELECT 'electricity_load', count(*) FROM silver.electricity_load UNION ALL
SELECT 'gtfs_routes', count(*) FROM silver.gtfs_routes UNION ALL
SELECT 'gtfs_departures', count(*) FROM silver.gtfs_departures
