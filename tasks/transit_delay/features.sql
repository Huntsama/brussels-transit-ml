-- The model table for transit_delay, built by DuckDB straight from the two Parquet files (features.py runs this).
-- Inputs: views slh_raw (stop_line_hourly) and city_raw (city_hourly); `hour` is UTC without a time zone.
-- Outputs: temp tables per_day (one row per local day) and features (one row per stop x line x hour, full days only).
-- Same rules as the original notebook (tests/test_transit_features.py checks the two agree on the 18-day snapshot).

-- local (Brussels) date, hour of day and weekday (0 = Monday) for every stop/line/hour
create or replace temp table base as
select stop_id, line_id, mode, hour, scheduled_wait_min, excess_wait_min,
       ((hour at time zone 'UTC') at time zone 'Europe/Brussels')::date         as date,
       hour(((hour at time zone 'UTC') at time zone 'Europe/Brussels'))         as hod,
       isodow(((hour at time zone 'UTC') at time zone 'Europe/Brussels')) - 1   as dow
from slh_raw;

-- days with fewer than 20 hours of data are partial (the collector starting up, today); a day is disrupted when
-- its median excess wait is more than 3x the normal one (the median of the full days' medians, at least 0.1 min).
-- Medians on the raw values, before clipping, as in the notebook.
create or replace temp table per_day as
with d as (
    select date, count(*) as rows, count(distinct hour) as hours, median(excess_wait_min) as median_excess
    from base group by date
), n as (select median(median_excess) as normal from d where hours >= 20)
select d.*, d.hours >= 20 as full_day,
       d.hours >= 20 and d.median_excess > 3 * greatest(n.normal, 0.1) as disrupted,
       n.normal as normal_median
from d, n
order by date;

create or replace temp table features as
with clean as (
    -- full days only; waits beyond +-30 min are feed glitches, clipped (CASE keeps NULL as NULL, unlike least())
    select b.stop_id, b.line_id, b.mode, b.hour, b.date, b.hod, b.dow, p.disrupted, b.scheduled_wait_min,
           case when b.excess_wait_min > 30 then 30.0 when b.excess_wait_min < -30 then -30.0
                else b.excess_wait_min end as excess_wait_min
    from base b join per_day p using (date)
    where p.full_day
), lagged as (
    select *,
           -- same stop/line exactly one hour earlier (the row before only counts if it is that hour)
           case when lag(hour) over w_sl = hour - interval 1 hour then lag(excess_wait_min) over w_sl end
               as prev_hour_ewt,
           -- mean over EARLIER rows of the same stop/line/hour of day, disrupted days left out of the history
           -- (= mlkit.past_mean: avg of NULLs is NULL when there is no history yet)
           avg(case when not disrupted then excess_wait_min end) over (
               partition by stop_id, line_id, hod order by hour
               rows between unbounded preceding and 1 preceding) as prev_days_same_hod_ewt
    from clean
    window w_sl as (partition by stop_id, line_id order by hour)
)
select l.stop_id, l.line_id, l.mode, l.hour, l.date, l.hod, l.dow, l.disrupted,
       l.scheduled_wait_min, l.excess_wait_min, l.prev_hour_ewt, l.prev_days_same_hod_ewt,
       -- the whole city one hour earlier (raw city table, not limited to full days)
       c.excess_wait_min as city_prev_ewt, c.no2 as no2_prev, c.load_mw as load_prev
from lagged l
left join city_raw c on c.hour = l.hour - interval 1 hour
order by l.hour, l.stop_id, l.line_id;
