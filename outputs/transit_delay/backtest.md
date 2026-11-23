# transit_delay: rolling backtest

Each fold trains on every day before it (disrupted days out), stops early on the 2 days before the test week, and is scored on the week (MAE, minutes; lower is better).

| Fold | Train days | Validation | Test week | Test rows |
|---|---|---|---|---|
| 1 | Sat 10 Oct -> Fri 13 Nov (34 days) | Sat 14 Nov, Sun 15 Nov | Mon 16 Nov -> Sun 22 Nov | 331,792 |
| 2 | Sat 10 Oct -> Fri 06 Nov (27 days) | Sat 07 Nov, Sun 08 Nov | Mon 09 Nov -> Sun 15 Nov | 336,419 |
| 3 | Sat 10 Oct -> Fri 30 Oct (20 days) | Sat 31 Oct, Sun 01 Nov | Mon 02 Nov -> Sun 08 Nov | 335,773 |

| Model | fold 1 | fold 2 | fold 3 | mean ± std |
|---|---|---|---|---|
| LightGBM without air/grid/city features | 0.910 | 0.882 | 0.882 | 0.892 ± 0.016 |
| LightGBM | 0.914 | 0.886 | 0.886 | 0.895 ± 0.016 |
| Average per stop/line/hour of day | 0.982 | 0.963 | 0.968 | 0.971 ± 0.010 |
| Always 0 (trust the timetable) | 1.228 | 1.182 | 1.170 | 1.193 ± 0.031 |
| Same as one hour ago | 1.318 | 1.274 | 1.250 | 1.281 ± 0.034 |

LightGBM vs the best simple rule of each fold:

- fold 1: 0.914 vs 0.982 (Average per stop/line/hour of day): **7.0% better**
- fold 2: 0.886 vs 0.963 (Average per stop/line/hour of day): **8.0% better**
- fold 3: 0.886 vs 0.968 (Average per stop/line/hour of day): **8.4% better**

Mean gain 7.8% (min 7.0%, max 8.4%) over 3 folds.

**Reading it.** LightGBM beat every simple rule in all 3 folds; its edge ranges from 7.0% to 8.4%. With 3 folds, the ± is a rough spread, not a confidence interval. Later folds train on more days, so part of a difference between folds is more history, part is the week itself (holidays, disruptions). The disrupted-day flag is decided once on all the data (median rule), so it is not a forecast of disruptions; disrupted days stay in the test weeks and are scored like any other day.
