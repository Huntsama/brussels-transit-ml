# Grid load: a fair comparison with Elia's day-ahead forecast

The main study (`notebook.py`) feeds the models the *measured* weather of day D (Open-Meteo archive, ERA5
reanalysis). Nobody has that at the end of D-1, when Elia publishes its day-ahead forecast, so the scores with
weather there are an upper bound. Here the same models are scored with weather *forecasts* as they were issued at
the time (Open-Meteo Previous Runs API, ECMWF IFS 0.25°, Brussels point, same four variables).

- Models, features, LightGBM settings and split are unchanged: trained on 2015-2022 and early-stopped on 2023, both
  with measured weather (no forecast archive exists that far back). Only the test inputs change.
- **Issued D-2** = the run ~48 h before each hour: all of it was out before Elia published. Strictly fair.
- **Issued D-1** = the run ~24 h before each hour: about Elia's timing, but the runs for afternoon and evening hours
  come out after Elia's morning publication, so slightly optimistic.
- Test window: **7 Mar 2024 to 21 Oct 2026 (UTC), 22,870 hours**, of which 767 on Belgian public holidays. It is the
  notebook's test period (from 1 Jan 2024) cut to the hours the ECMWF forecast archive covers (it starts in March
  2024 and has a gap in April 2026). Elia's MAE on this window is 259 MW (256 on the full notebook window).

Produced by `tasks/grid_load/fair_comparison.py` (numbers in `fair-comparison.csv`). Run of 15 Nov 2026.

| Model | MAE (MW) | vs Elia | Holidays MAE (MW) | Holidays vs Elia |
|---|---|---|---|---|
| Elia day-ahead forecast | 258.8 | 0 | 322.4 | 0 |
| LightGBM, no weather | 288.6 | +11.5% | 281.2 | -12.8% |
| LightGBM, measured weather (upper bound) | 266.8 | +3.1% | 244.5 | -24.2% |
| LightGBM, forecast weather issued D-1 | 262.8 | +1.5% | 242.2 | -24.9% |
| LightGBM, forecast weather issued D-2 | 264.5 | +2.2% | 242.9 | -24.6% |
| LightGBM + Elia as input, measured weather (upper bound) | 238.7 | -7.8% | 253.7 | -21.3% |
| LightGBM + Elia as input, forecast weather issued D-1 | 239.8 | -7.3% | 253.1 | -21.5% |
| **LightGBM + Elia as input, forecast weather issued D-2** | **241.2** | **-6.8%** | **257.2** | **-20.2%** |

Check with Open-Meteo's default model mix (`best_match`) instead of ECMWF, same window: LightGBM + Elia 243.3 MW
(D-1, -6.0%) and 240.6 MW (D-2, -7.0%); LightGBM alone 271.3 (D-1) and 264.8 (D-2). Its D-1 runs read about 1.2 °C
warmer than the archive since mid-2024 (its D-2 runs do not), which is why its D-1 row is the worst; ECMWF has no such
bias, so it is the main source.

**What this says.** The audit was right that the old 235 vs 256 MW compared our measured-weather model with Elia's
forecast-weather one. With weather forecasts that were really available in time, the correction model still beats
Elia: **241 vs 259 MW, 6.8% better** (strict D-2), 7.3% with D-1 runs. Swapping measured weather for forecasts costs the
correction model only 1-3 MW (upper bound 238.7). Weather is worth about 22 MW to LightGBM in total (288.6 without it),
and a day-ahead ECMWF forecast keeps all of that; on this window the forecast-fed model is even slightly better
than the measured-weather one, which is within noise. On its own, without Elia's forecast, LightGBM does *not* beat Elia overall (263-265 vs
259 MW), but it does on public holidays (243 vs 322 MW, about 25% better).

**Limits, said plainly.** One test window, no confidence interval; the 2015-2023 training rows still use measured
weather because no forecast archive goes back that far; one weather point (Brussels) for the whole country; the
model choices were made while looking at the full 2024-2026 test period in the notebook, so the test is not fully
blind.
