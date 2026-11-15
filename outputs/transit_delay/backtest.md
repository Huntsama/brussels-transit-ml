# transit_delay: rolling backtest

Each fold trains on every day before it (disrupted days out), stops early on the 2 days before the test week, and is scored on the week (MAE, minutes; lower is better).

| Fold | Train days | Validation | Test week | Test rows |
|---|---|---|---|---|
| 1 | Sat 10 Oct -> Thu 05 Nov (26 days) | Fri 06 Nov, Sat 07 Nov | Sun 08 Nov -> Sat 14 Nov | 336,346 |
| 2 | Sat 10 Oct -> Thu 29 Oct (19 days) | Fri 30 Oct, Sat 31 Oct | Sun 01 Nov -> Sat 07 Nov | 335,891 |

| Model | fold 1 | fold 2 | mean ± std |
|---|---|---|---|
| LightGBM without air/grid/city features | 0.888 | 0.881 | 0.884 ± 0.005 |
| LightGBM | 0.892 | 0.885 | 0.888 ± 0.005 |
| Average per stop/line/hour of day | 0.970 | 0.970 | 0.970 ± 0.000 |
| Always 0 (trust the timetable) | 1.185 | 1.171 | 1.178 ± 0.010 |
| Same as one hour ago | 1.275 | 1.252 | 1.263 ± 0.017 |

LightGBM vs the best simple rule of each fold:

- fold 1: 0.892 vs 0.970 (Average per stop/line/hour of day): **8.1% better**
- fold 2: 0.885 vs 0.970 (Average per stop/line/hour of day): **8.7% better**

Mean gain 8.4% (min 8.1%, max 8.7%) over 2 folds.

**Reading it.** LightGBM beat every simple rule in all 2 folds; its edge ranges from 8.1% to 8.7%. With 2 folds, the ± is a rough spread, not a confidence interval. Later folds train on more days, so part of a difference between folds is more history, part is the week itself (holidays, disruptions). The disrupted-day flag is decided once on all the data (median rule), so it is not a forecast of disruptions; disrupted days stay in the test weeks and are scored like any other day.
