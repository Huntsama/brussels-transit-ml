# Brussels transit ML

[![ci](https://github.com/Huntsama/brussels-transit-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/Huntsama/brussels-transit-ml/actions/workflows/ci.yml)
[![weekly retrain](https://github.com/Huntsama/brussels-transit-ml/actions/workflows/retrain.yml/badge.svg)](https://github.com/Huntsama/brussels-transit-ml/actions/workflows/retrain.yml)

Machine learning on Belgian open data: **correcting the grid operator's electricity forecast** and **predicting how
late Brussels buses, trams and metros run**, on data my own [lakehouse](https://github.com/Huntsama/brussels-lakehouse)
collects every 5 minutes. Every model is tested on later data than it was trained on, against honest baselines.

## Beating Elia's day-ahead forecast

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/readme-elia-dark.png">
  <img alt="Mean absolute error: Elia 259 MW, LightGBM alone 264 MW, LightGBM correcting Elia 241 MW; on public holidays 322, 243 and 257 MW" src="docs/img/readme-elia-light.png">
</picture>

LightGBM takes Elia's own forecast as one input and corrects it, using only weather forecasts issued two days ahead,
so it knows nothing Elia did not. Over 22,870 unseen hours it is **7% more accurate than Elia**. On its own it does
not beat Elia overall, but it does on public holidays, where Elia's error jumps. Method and limits:
[fair comparison](outputs/grid_load/fair-comparison.md).

## Predicting STIB delays, week after week

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/readme-backtest-dark.png">
  <img alt="Per test week, mean absolute error: trust the timetable about 1.18 min, best simple rule 0.97, LightGBM 0.88 to 0.89" src="docs/img/readme-backtest-light.png">
</picture>

For every stop and line, how much longer than the timetable will people wait next hour? On 1.7M stop-hours, LightGBM
beats the best simple rule (an average per stop, line and hour) by **8.5%**, and by 8.5% and 9.2% on two separate
test weeks. It retrains every Monday on fresh data; the run fails, and I get an email, if it ever stops beating the
rule by 3%. Air quality and grid load add nothing once the daily rhythm is known, so the final model leaves them out.

## What the data shows

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/readme-strike-dark.png">
  <img alt="Daily average extra wait over five weeks: about 0.4 minutes most days, 4.8 minutes on the 12 October strike" src="docs/img/readme-strike-light.png">
</picture>


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/img/readme-heatmap-dark.png">
  <img alt="Extra wait by weekday and hour: highest in the weekday afternoon rush, peaking Friday 17:00 at about 1 minute" src="docs/img/readme-heatmap-light.png">
</picture>


## How it works

```mermaid
flowchart LR
  L[(lakehouse<br/>STIB, air, grid<br/>every 5-15 min)] -->|weekly export| F[features.sql<br/>DuckDB]
  F --> T[train<br/>LightGBM + NN]
  T --> E[evaluate +<br/>backtest]
  E --> H{gain over<br/>best rule >= 3%?}
  H -->|yes| C[commit results<br/>history.csv]
  H -->|no| A[run fails,<br/>email alert]
  O[Elia + Open-Meteo<br/>archived forecasts] --> G[grid_load:<br/>LightGBM correcting Elia]
```

Python, DuckDB SQL, LightGBM, PyTorch, MLflow, pytest, GitHub Actions. Tests check that no feature ever sees the
future and that the SQL features match the original pandas ones exactly.

```bash
uv sync
bash run.sh grid_load                 # or: FETCH=0 bash run.sh transit_delay
uv run pytest -q tests
```

## More

- [docs/DETAILS.md](docs/DETAILS.md): full results tables, the Databricks platform (Terraform, Unity Catalog, written
  but not yet deployed), live status with Kafka and FastAPI, cost, deployment, ideas.
- [docs/EXPLAINED.md](docs/EXPLAINED.md): the whole project in plain words, with diagrams.
- Full studies with every plot: `outputs/<task>/notebook.ipynb`.
