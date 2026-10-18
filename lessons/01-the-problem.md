# Lesson 1: the problem and the training table

## The question

> For a given stop, line and hour, how many minutes longer than the timetable will people wait?

That number is `excess_wait_min` in `brussels.gold.stop_line_hourly`. Predicting it ahead of time is useful: an app
could say "line 46 at Montgomery usually runs 2 minutes behind at 17:00", and STIB could see which stops are going
to have a bad hour.

Three words to own before writing any code:

- **Target** (`y`): what the model predicts. Here `excess_wait_min`, a number, so this is **regression**.
- **Unit**: what one row is. Here one stop × line × hour. 320,000 rows for the first week.
- **Features** (`X`): what the model is allowed to look at to make the guess.

## Leakage: the mistake that makes models look brilliant and fail in real life

A feature leaks when it would not be known at the moment you make the prediction. `actual_wait_min` is the obvious
one: excess wait is `actual - scheduled`, so with `actual_wait_min` as a feature the model "predicts" perfectly and
is useless. Less obvious: the average excess wait of *this* stop over the *whole* week, including the hour you are
predicting. That average was computed with the answer in it.

The rule: for every feature, ask "at 07:00 on Friday, predicting 08:00, would I have this number?"

## Which features are safe

| Feature | Where from | Safe? |
|---|---|---|
| `line_id`, `mode`, `stop_id` | the row | yes, known in advance |
| hour of day, weekday | `hour` | yes |
| `scheduled_wait_min` | timetable | yes, published weeks ahead |
| excess wait of this stop and line one hour earlier | `lag()` over the table | yes, if you predict one hour ahead |
| same stop, line and hour of day, previous days | window over earlier rows only | yes |
| `actual_wait_min`, `snapshots` | the row | **no**, measured during the hour |
| NO₂, grid load in that hour | `gold.city_hourly` | think about it (question 3) |

## Your task

In the Git folder (`/Repos/<you>/brussels-transit-ml`), create the notebook `lessons/01_training_table` and write a
SQL cell that builds `brussels.ml.training` with one row per stop × line × hour and these columns:

1. `stop_id`, `line_id`, `mode`, `hour`, and the target `excess_wait_min`
2. `hod` (hour of day, Brussels time) and `dow` (weekday). Hint: `from_utc_timestamp(hour, 'Europe/Brussels')`
3. `scheduled_wait_min`
4. `prev_hour_ewt`: the excess wait of the same stop and line exactly one hour earlier, or null.
   Hint: `lag()` is not enough when hours are missing; join the table to itself on `hour - INTERVAL 1 HOUR`.
5. `prev_days_same_hod_ewt`: the average excess wait of the same stop, line and hour of day on **earlier days only**.
   Hint: a window `PARTITION BY stop_id, line_id, hod ORDER BY hour ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING`
6. `is_disrupted_day`: true for Monday 12 Oct 2026 (the bad day from the lakehouse README). You will use it to
   leave that day out of training, or test on it alone.

Then check your table:

```sql
SELECT count(*), count(prev_hour_ewt), count(prev_days_same_hod_ewt) FROM brussels.ml.training;
```

## Done when

- `brussels.ml.training` exists, with as many rows as `gold.stop_line_hourly`
- on Monday's rows `prev_days_same_hod_ewt` is null (there is no earlier day yet)
- you can answer the questions below out loud

## Questions

1. Why is `snapshots` a leak, even though it doesn't contain the answer directly?
2. `prev_days_same_hod_ewt` on Friday averages Mon to Thu. Monday was a disrupted day. What does that do to Friday's
   feature, and what would you do about it?
3. Air quality in the same hour: leak or not? Does it depend on whether you predict one hour ahead or one day ahead?
4. You have one week. Which pattern can't the model learn yet, however good it is?
