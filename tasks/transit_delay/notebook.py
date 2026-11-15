# %% [markdown]
# # Brussels transit: what drives the wait, and can we predict it?
#
# **Data:** the Brussels Lakehouse (Azure ADLS Gen2). STIB live waiting times every 5 minutes, air quality (IRCEL)
# and Belgian grid load (Elia) every 15 minutes, cleaned by dbt into two gold tables:
#
# - `stop_line_hourly`: one row per stop × line × hour, with `excess_wait_min` = how many minutes longer people wait
#   than the timetable promises.
# - `city_hourly`: one row per hour for the whole city: excess wait, NO₂, PM10, PM2.5, grid load.
#
# **Questions**
# 1. Is the data complete and clean?
# 2. Does the wait move with air quality or electricity load, or is that just rush hour?
# 3. Can a model predict next hour's excess wait at each stop better than simple rules?
#    Baselines vs LightGBM vs a neural network.
#
# **This notebook is the report, not the computation.** With five weeks of data (1.7 M stop-line-hours) doing
# everything in one Jupyter kernel ran out of memory, so the heavy work is split into steps that `bash run.sh
# transit_delay` runs first (settings in `tasks/transit_delay/config.yaml`):
#
# | Step | What | Output |
# |---|---|---|
# | `features.py` + `features.sql` | DuckDB builds the model table straight from the Parquet files | `data/transit_delay/features.parquet`, `per_day.csv` |
# | `train.py --nn` | baselines, LightGBM, neural network on the standard split; MLflow + model registry | `outputs/transit_delay/test_predictions.parquet`, `models/` |
# | `evaluate.py` | test MAE of every model, result figures | `metrics.csv`, `figures/07..10` |
# | `backtest.py` | the same models on each of the last weeks in turn | `backtest.csv`, `backtest.md` |
#
# This notebook only reads those files (about 150 MB of memory) and explains them.

# %%
import json, sys, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "src" / "mlkit.py").exists():
        sys.path.insert(0, str(p / "src")); sys.path.insert(0, str(p / "tasks" / "transit_delay")); break
import mlkit

import duckdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, Markdown, display
from common import load_config, load_features

DATA, OUT = mlkit.paths("transit_delay")
assert (DATA / "features.parquet").exists() and (OUT / "metrics.csv").exists(), \
    "run the pipeline first: bash run.sh transit_delay"
plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
cfg = load_config()
TARGET, FEATURES = cfg["target"], cfg["features"]
meta = json.loads((DATA / "features.meta.json").read_text())
summary = json.loads((OUT / "train_summary.json").read_text())


def save(name):
    mlkit.save_fig(OUT, name)


def show(name):
    display(Image(filename=str(OUT / "figures" / f"{name}.png")))


# %%
raw = f"read_parquet('{DATA / 'stop_line_hourly.parquet'}')"
print(duckdb.sql(f"select count(*) as rows, count(distinct stop_id) as stops, count(distinct line_id) as lines, "
                 f"min(hour) as first_hour_utc, max(hour) as last_hour_utc from {raw}").df().to_string(index=False))
print("data snapshot id:", meta["snapshot_id"])
duckdb.sql(f"select * exclude (stop_id, line_id, mode, hour) from {raw}").df().describe().round(2)

# %% [markdown]
# ## 1. Is the data complete and clean?

# %%
per_day = pd.read_csv(DATA / "per_day.csv", parse_dates=["date"]).set_index("date")
print(per_day[["rows", "hours", "median_excess", "full_day", "disrupted"]].round(2).to_string())
nulls = duckdb.sql(f"select count(*) - count(columns(*)) from {raw}").df().iloc[0]
print("\nMissing values per column:\n", nulls[nulls > 0].to_string() or "none")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
labels = per_day.index.strftime("%d %b")
ax[0].bar(labels, per_day["hours"], color="#6c8ebf"); ax[0].set_title("Hours with data per day (24 = complete)")
ax[1].bar(labels, per_day["median_excess"], color="#d97757"); ax[1].set_title("Median excess wait per day (min)")
for a in ax: a.tick_params(axis="x", labelrotation=90, labelsize=7)
save("01_completeness")

# %% [markdown]
# **Cleaning rules (in `features.sql`)**
# - Days with fewer than 20 hours of data are partial (the collector starting up, or today): left out.
# - A *disrupted day* is one whose median excess wait is more than 3× the normal median. It is real data, so it
#   is not deleted, but it is left out of **training** (a model should learn normal service) and reported apart.
# - Excess waits beyond ±30 min are prediction glitches in STIB's feed (a vehicle cancelled, then reappearing);
#   they are clipped so one glitch cannot dominate the error.

# %%
full = per_day[per_day.full_day]
print("normal daily median excess wait:", round(full["median_excess"].median(), 2), "min")
print("disrupted days:", [f"{d:%a %d %b}" for d in full.index[full.disrupted]])
print("partial days left out:", [f"{d:%a %d %b}" for d in per_day.index[~per_day.full_day]])
n_clip = duckdb.sql(f"select count(*) from {raw} where abs(excess_wait_min) > 30").fetchone()[0]
print(f"clipped {n_clip:,} glitch values ({n_clip / meta['feature_rows']:.3%}); {meta['feature_rows']:,} rows kept")

# %% [markdown]
# ## 2. When is the wait longest?

# %%
df = load_features()  # the model table: compact types, ~100 MB in memory
normal_days = df[~df.disrupted]
heat = normal_days.pivot_table(index="dow", columns="hod", values=TARGET, aggfunc="median")
plt.figure(figsize=(12, 3.2))
plt.imshow(heat, aspect="auto", cmap="YlOrRd")
plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(heat)])
plt.xticks(range(len(heat.columns)), heat.columns); plt.xlabel("hour of day (Brussels)")
plt.colorbar(label="median excess wait (min)"); plt.title("Excess wait by weekday and hour (normal days)")
plt.grid(False); save("02_week_heatmap")

by_mode = normal_days.groupby(["mode", "hod"], observed=True)[TARGET].median().unstack(0)
by_mode.plot(title="Median excess wait by hour of day and mode", ylabel="min"); save("03_mode_by_hour")
del normal_days

# %% [markdown]
# ## 3. Air quality, electricity and the wait: correlation
#
# **The trap:** rush hour raises traffic, NO₂, power use *and* transit delays at the same time. A raw correlation
# mostly measures that shared daily rhythm. So each measure is also correlated **within the hour of day**: subtract
# that hour's average first, and ask "when NO₂ is higher *than usual for 08:00*, is the wait also higher than
# usual for 08:00?" That is the number that could mean a real link. Even then: correlation is not causation.

# %%
city = pd.read_parquet(DATA / "city_hourly.parquet")
city["hour"] = pd.to_datetime(city["hour"], utc=True)
local = city["hour"].dt.tz_convert("Europe/Brussels")
city["date"], city["hod"] = pd.to_datetime(local.dt.date), local.dt.hour
MEASURES = ["no2", "pm10", "pm25", "load_mw", "vehicles_avg", "median_wait_min"]
c = city.dropna(subset=["excess_wait_min"])
c = c[c["date"].isin(full.index)]
deseason = c[MEASURES + ["excess_wait_min"]] - c.groupby("hod")[MEASURES + ["excess_wait_min"]].transform("mean")
corr = pd.DataFrame({
    "r_raw": c[MEASURES].corrwith(c["excess_wait_min"]),
    "r_within_hour": deseason[MEASURES].corrwith(deseason["excess_wait_min"]),
    "hours": c[MEASURES].notna().sum(),
}).round(2)
print(corr.to_string())

fig, ax = plt.subplots(figsize=(8, 3.5))
corr[["r_raw", "r_within_hour"]].plot.barh(ax=ax, color=["#bbbbbb", "#5b3df5"])
ax.axvline(0, color="k", lw=0.8); ax.set_xlim(-1, 1); ax.set_title("Correlation with city excess wait")
save("04_correlation")

# %%
fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
ax[0].plot(c["hour"], c["excess_wait_min"], color="#d97757"); ax[0].set_ylabel("excess wait (min)")
ax[1].plot(c["hour"], c["no2"], color="#6c8ebf"); ax[1].set_ylabel("NO₂ (µg/m³)")
ax[2].plot(c["hour"], c["load_mw"], color="#57ab5a"); ax[2].set_ylabel("grid load (MW)")
ax[0].set_title("The city, hour by hour"); save("05_timeseries")

# %%
# Does a measure lead the wait? Within-hour correlation with the measure shifted 0 to 6 hours earlier.
lags = {m: [deseason["excess_wait_min"].corr(deseason[m].shift(k)) for k in range(7)] for m in ["no2", "load_mw"]}
pd.DataFrame(lags, index=[f"{k}h before" for k in range(7)]).plot(marker="o", title="Lagged within-hour correlation")
plt.axhline(0, color="k", lw=0.8); save("06_lagged_correlation")

# %% [markdown]
# ## 4. The prediction task
#
# Predict **next hour's excess wait** for every stop × line, using only what is known an hour before.
#
# | Feature | Why it is allowed |
# |---|---|
# | stop, line, mode, hour of day, weekday | known in advance |
# | scheduled wait | published timetable |
# | excess wait of this stop/line one hour earlier | already measured |
# | average excess wait of this stop/line/hour-of-day on **earlier days only** | no future data inside |
# | city excess wait, NO₂, grid load one hour earlier | already measured |
#
# Not allowed: `actual_wait_min` (it *is* the answer) and `snapshots` (only known once the hour is over).
# The features are computed in SQL (`features.sql`); a test checks they equal the original pandas code on the
# 18-day snapshot and that changing later data never changes an earlier row's features.
#
# **Split by time, never at random:** a random split puts Thursday 08:05 in training and 08:10 in the test, so the
# model "remembers" instead of predicting. Training = the first days, test = the last quarter of the days, the two
# days before the test = validation (early stopping).

# %%
s = summary
print(f"train {s['rows']['train']:,} rows ({s['train_days'][0]} -> {s['train_days'][1]}), "
      f"val {s['rows']['val']:,} ({', '.join(s['val_days'])}), test {s['rows']['test']:,} "
      f"({s['test_days'][0]} -> {s['test_days'][1]})")
test_rows = df[(df.date >= s["test_days"][0]) & (df.date <= s["test_days"][1])]
print("feature coverage in test:\n", test_rows[FEATURES].notna().mean().round(2).to_string())
del test_rows

# %% [markdown]
# ## 5. Baselines: the numbers any model has to beat
# Error = **MAE**, the average of |predicted − actual| in minutes. Easy to explain, not dominated by glitches.
#
# - *Always 0*: trust the timetable.
# - *Average per stop/line/hour of day*: the mean excess wait of that stop, line and hour over the training days.
# - *Same as one hour ago*: this stop/line's excess wait in the previous hour.
#
# ## 6. LightGBM (gradient-boosted trees)
# Builds hundreds of small decision trees, each correcting the errors of the ones before. Handles categories
# (stop, line) and missing values natively; usually the strongest model on tables like this. Fitted twice: with all
# features, and without the air/grid/city ones, to see whether those help.
#
# ## 7. A neural network
# A multilayer perceptron with **embeddings**: each stop, line, hour and weekday gets a small learned vector
# (like word embeddings in NLP), concatenated with the numeric features and passed through two hidden layers.
# Trained with the L1 loss (= MAE) and early stopping on the validation days. CPU is enough (`train.py --nn`).

# %%
for name, m in summary["models"].items():
    extra = f"{m['trees']} trees" if "trees" in m else f"{m['epochs']} epochs"
    print(f"{name}: {extra}, {m['fit_seconds']}s")
show("07_importance")
if (OUT / "figures" / "08_nn_training.png").exists() and "Neural network (embeddings + MLP)" in summary["models"]:
    show("08_nn_training")

# %% [markdown]
# ## 8. Results (standard split)

# %%
metrics = pd.read_csv(OUT / "metrics.csv", index_col=0)
print(metrics.round(3).to_string())
show("09_results")

# %% [markdown]
# ## 8b. MLOps: tracked in MLflow, best LightGBM registered as `transit_delay_lgbm`
# `train.py` logs every model's test MAE, the data snapshot id (sha256 of the input Parquet files + first/last hour)
# and the git commit, then registers the LightGBM that did better **on the validation days**, promoted to
# `champion` only if it beats the current one on the same test rows.

# %%
print({k: summary.get(k) for k in ["snapshot_id", "git_commit", "registered", "registry"]})

# %% [markdown]
# ## 9. Where is the model wrong?

# %%
show("10_error_breakdown")
print("Hardest lines to predict (LightGBM, test):\n", pd.read_csv(OUT / "hardest_lines.csv", index_col=0).to_string())

# %% [markdown]
# ## 9b. Is one test period enough? Rolling backtest
# One split can be lucky. `backtest.py` repeats the comparison on each recent week: train on everything before it,
# validate on the two days before it, test on the week.

# %%
display(Markdown((OUT / "backtest.md").read_text()))

# %% [markdown]
# ## 10. What we found (run of 15 Nov 2026, data 10 Oct to 14 Nov)
#
# **Data.** 1,729,314 stop-line-hours, 1,894 stops, 89 lines, no missing values. 35 full days (Fri 16 Oct and
# today, Sun 15 Nov, are partial and left out); one disrupted day (Mon 12 Oct: median excess wait 4.44 min against
# a normal 0.34 min), kept out of training. 361 glitch values clipped (0.02%). 1,659,005 rows in the model table.
#
# **Correlation.** With five weeks, the links with the city measures are weaker than in the first 18 days: within
# the hour of day, vehicles on the road r = −0.14 (was −0.29), grid load +0.14 (+0.17), NO₂ +0.05 (+0.19).
# None of them is strong enough to matter for a prediction, and the model confirms it (below).
#
# **Prediction, standard split (test = Sat 07 to Sat 14 Nov, 385k rows, never seen in training).**
#
# | Model | MAE (min), 5 weeks | 18 days (test 23-26 Oct) |
# |---|---|---|
# | Same as one hour ago | 1.290 | 1.374 |
# | Always 0 (trust the timetable) | 1.194 | 1.277 |
# | Average per stop/line/hour of day | 0.987 | 1.146 |
# | Neural network (embeddings + MLP) | 0.921 | 1.053 |
# | LightGBM | 0.910 | 1.027 |
# | **LightGBM without air/grid/city features** | **0.903** | **1.018** |
#
# - LightGBM still wins, but **its edge over the best simple rule shrank from 11% to 8.5%** (0.987 → 0.903).
#   Most of the change is the rule getting better: with five weeks of history, "the average of this stop, line and
#   hour" is a much better guess (1.146 → 0.987), and that average is also LightGBM's main input.
# - The **rolling backtest** agrees: on the two most recent weeks LightGBM is 8.0% and 8.7% better than the best
#   rule (0.893 vs 0.970, 0.885 vs 0.970). Only two weeks have the 14 training days the backtest asks for, so the
#   spread between folds says little.
# - Air quality, grid load and city delays still do not help: without them the model is slightly better, on the
#   standard split and in both backtest weeks.
# - The model without city features used all 2,000 trees in every run (early stopping never triggered), so it is
#   not fully converged; more trees or a higher learning rate may gain a little more.
# - The neural network is close (0.921) but trees still win on this table.
# - Hardest to predict: line 72 (3.2 min mean error) and the night buses (N04 to N18): few vehicles per hour, so
#   one late bus moves the average a lot.
#
# **Next.** A weather feature, a disruption detector that flags a bad day as it starts (the median rule here uses
# the whole day), and the second target: correcting STIB's own countdown.
