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
# Runs on a laptop CPU in a few minutes. In Colab: upload the two Parquet files from `study/data/` when asked.

# %%
import os, sys, time, warnings
warnings.filterwarnings("ignore")

from pathlib import Path
# src/mlkit.py from the repo; in Colab, upload mlkit.py next to the notebook
for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "src" / "mlkit.py").exists():
        sys.path.insert(0, str(p / "src")); break
import mlkit
if mlkit.IN_COLAB:
    os.system("pip -q install lightgbm")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
import torch
from torch import nn

DATA, OUT = mlkit.paths("transit_delay")
mlkit.colab_upload(DATA, ["stop_line_hourly.parquet", "city_hourly.parquet"])
plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})
torch.manual_seed(0); np.random.seed(0)
TZ = "Europe/Brussels"


def save(name):
    mlkit.save_fig(OUT, name)


# %%
slh = pd.read_parquet(DATA / "stop_line_hourly.parquet")
city = pd.read_parquet(DATA / "city_hourly.parquet")
for df in (slh, city):
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    local = df["hour"].dt.tz_convert(TZ)
    df["date"] = local.dt.date
    df["hod"] = local.dt.hour
    df["dow"] = local.dt.dayofweek  # 0 = Monday
print(f"stop_line_hourly: {len(slh):,} rows, {slh.stop_id.nunique():,} stops, {slh.line_id.nunique()} lines")
print(f"from {slh.hour.min():%a %d %b %H:%M} to {slh.hour.max():%a %d %b %H:%M} UTC")
slh.describe().round(2)

# %% [markdown]
# ## 1. Is the data complete and clean?

# %%
per_day = slh.groupby("date").agg(rows=("excess_wait_min", "size"), hours=("hour", "nunique"),
                                  median_excess=("excess_wait_min", "median"))
print(per_day.round(2).to_string())
print("\nMissing values per column:\n", slh.isna().sum()[lambda s: s > 0].to_string() or "none")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
per_day["hours"].plot.bar(ax=ax[0], color="#6c8ebf", title="Hours with data per day (24 = complete)")
per_day["median_excess"].plot.bar(ax=ax[1], color="#d97757", title="Median excess wait per day (min)")
for a in ax: a.tick_params(axis="x", labelrotation=60)
save("01_completeness")

# %% [markdown]
# **Cleaning rules used below**
# - Days with fewer than 20 hours of data are partial (the collector starting up): left out.
# - A *disrupted day* is one whose median excess wait is more than 3× the normal median. It is real data, so it
#   is not deleted, but it is left out of **training** (a model should learn normal service) and reported apart.
# - Excess waits beyond ±30 min are prediction glitches in STIB's feed (a vehicle cancelled, then reappearing);
#   they are clipped so one glitch cannot dominate the error.

# %%
full_days = per_day.index[per_day["hours"] >= 20]
normal = per_day.loc[full_days, "median_excess"].median()
disrupted = [d for d in full_days if per_day.loc[d, "median_excess"] > 3 * max(normal, 0.1)]
print("normal daily median excess wait:", round(normal, 2), "min")
print("disrupted days:", [f"{d:%a %d %b}" for d in disrupted])

n_clip = (slh["excess_wait_min"].abs() > 30).sum()
slh = slh[slh["date"].isin(full_days)].copy()
slh["excess_wait_min"] = slh["excess_wait_min"].clip(-30, 30)
slh["disrupted"] = slh["date"].isin(disrupted)
print(f"clipped {n_clip:,} glitch values ({n_clip / len(slh):.3%}); {len(slh):,} rows kept")

# %% [markdown]
# ## 2. When is the wait longest?

# %%
heat = slh[~slh.disrupted].pivot_table(index="dow", columns="hod", values="excess_wait_min", aggfunc="median")
plt.figure(figsize=(12, 3.2))
plt.imshow(heat, aspect="auto", cmap="YlOrRd")
plt.yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][: len(heat)])
plt.xticks(range(len(heat.columns)), heat.columns); plt.xlabel("hour of day (Brussels)")
plt.colorbar(label="median excess wait (min)"); plt.title("Excess wait by weekday and hour (normal days)")
plt.grid(False); save("02_week_heatmap")

by_mode = slh[~slh.disrupted].groupby(["mode", "hod"])["excess_wait_min"].median().unstack(0)
by_mode.plot(title="Median excess wait by hour of day and mode", ylabel="min"); save("03_mode_by_hour")

# %% [markdown]
# ## 3. Air quality, electricity and the wait: correlation
#
# **The trap:** rush hour raises traffic, NO₂, power use *and* transit delays at the same time. A raw correlation
# mostly measures that shared daily rhythm. So each measure is also correlated **within the hour of day**: subtract
# that hour's average first, and ask "when NO₂ is higher *than usual for 08:00*, is the wait also higher than
# usual for 08:00?" That is the number that could mean a real link. Even then: correlation is not causation.

# %%
MEASURES = ["no2", "pm10", "pm25", "load_mw", "vehicles_avg", "median_wait_min"]
c = city.dropna(subset=["excess_wait_min"]).copy()
c = c[c["date"].isin(full_days)]
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
#
# **Split by time, never at random:** a random split puts Thursday 08:05 in training and 08:10 in the test, so the
# model "remembers" instead of predicting. Training = the first days, test = the last days.

# %%
df = slh.sort_values("hour").copy()
key = ["stop_id", "line_id"]
prev = df[key + ["hour", "excess_wait_min"]].copy()
prev["hour"] += pd.Timedelta(hours=1)
df = df.merge(prev.rename(columns={"excess_wait_min": "prev_hour_ewt"}), on=key + ["hour"], how="left")

# expanding mean of earlier days only (shift(1) drops the current row); disrupted days left out of the history
df = df.sort_values("hour").reset_index(drop=True)
hist = df["excess_wait_min"].where(~df["disrupted"])
groups = [df["stop_id"], df["line_id"], df["hod"]]
df["prev_days_same_hod_ewt"] = mlkit.past_mean(hist, groups)  # earlier rows only, tested in tests/

cprev = city[["hour", "excess_wait_min", "no2", "load_mw"]].copy()
cprev["hour"] += pd.Timedelta(hours=1)
df = df.merge(cprev.rename(columns={"excess_wait_min": "city_prev_ewt", "no2": "no2_prev", "load_mw": "load_prev"}),
              on="hour", how="left")

for col in ["stop_id", "line_id", "mode"]:
    df[col] = df[col].astype("category")

FEATURES = ["stop_id", "line_id", "mode", "hod", "dow", "scheduled_wait_min",
            "prev_hour_ewt", "prev_days_same_hod_ewt", "city_prev_ewt", "no2_prev", "load_prev"]
CITY_FEATURES = ["city_prev_ewt", "no2_prev", "load_prev"]
TARGET = "excess_wait_min"

days = sorted(df["date"].unique())
n_test = max(4, len(days) // 4)
test_days, val_days = days[-n_test:], days[-n_test - 2:-n_test]
train = df[~df["date"].isin(test_days + val_days) & ~df["disrupted"]]
val = df[df["date"].isin(val_days) & ~df["disrupted"]]
test = df[df["date"].isin(test_days)]
print(f"train {len(train):,} rows ({train.date.min():%d %b} -> {train.date.max():%d %b}), "
      f"val {len(val):,} ({val_days[0]:%d %b}, {val_days[-1]:%d %b}), test {len(test):,} "
      f"({test_days[0]:%a %d %b} -> {test_days[-1]:%a %d %b})")
print("feature coverage in test:\n", test[FEATURES].notna().mean().round(2).to_string())

# %% [markdown]
# ## 5. Baselines: the numbers any model has to beat
# Error = **MAE**, the average of |predicted − actual| in minutes. Easy to explain, not dominated by glitches.

# %%
results = {}


def mae(pred, y=test[TARGET]):
    return float(np.abs(np.asarray(pred) - y.values).mean())


results["Always 0 (trust the timetable)"] = mae(np.zeros(len(test)))
hod_mean = train.groupby(["stop_id", "line_id", "hod"], observed=True)[TARGET].mean().rename("p").reset_index()
p = test[["stop_id", "line_id", "hod"]].merge(hod_mean, how="left", on=["stop_id", "line_id", "hod"])["p"]
results["Average per stop/line/hour of day"] = mae(p.fillna(train[TARGET].median()))
results["Same as one hour ago"] = mae(test["prev_hour_ewt"].fillna(0))
for k, v in results.items():
    print(f"{v:6.3f} min  {k}")

# %% [markdown]
# ## 6. LightGBM (gradient-boosted trees)
# Builds hundreds of small decision trees, each correcting the errors of the ones before. Handles categories
# (stop, line) and missing values natively; usually the strongest model on tables like this.

# %%
def fit_lgb(features):
    m = lgb.LGBMRegressor(objective="l1", n_estimators=2000, learning_rate=0.05, num_leaves=63,
                          min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    m.fit(train[features], train[TARGET], eval_set=[(val[features], val[TARGET])],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


t = time.time()
gbm = fit_lgb(FEATURES)
results["LightGBM"] = mae(gbm.predict(test[FEATURES]))
print(f"LightGBM: {results['LightGBM']:.3f} min, {gbm.best_iteration_} trees, {time.time() - t:.0f}s")

no_city = [f for f in FEATURES if f not in CITY_FEATURES]
results["LightGBM without air/grid/city features"] = mae(fit_lgb(no_city).predict(test[no_city]))
print(f"without air/grid/city: {results['LightGBM without air/grid/city features']:.3f} min")

imp = pd.Series(gbm.booster_.feature_importance("gain"), index=FEATURES).sort_values()
(imp / imp.sum()).plot.barh(color="#5b3df5", title="What LightGBM relies on (share of gain)"); save("07_importance")

# %% [markdown]
# ## 7. A neural network
# A multilayer perceptron with **embeddings**: each stop, line, hour and weekday gets a small learned vector
# (like word embeddings in NLP), concatenated with the numeric features and passed through two hidden layers.
# Trained with the L1 loss (= MAE) and early stopping on the validation days. CPU is enough.

# %%
CATS = ["stop_id", "line_id", "mode", "hod", "dow"]
NUMS = ["scheduled_wait_min", "prev_hour_ewt", "prev_days_same_hod_ewt", "city_prev_ewt", "no2_prev", "load_prev"]
codes = {c: {v: i + 1 for i, v in enumerate(train[c].astype(str).unique())} for c in CATS}  # 0 = unseen
mu, sd = train[NUMS].mean(), train[NUMS].std().replace(0, 1)


def tensors(d):
    x_cat = np.stack([d[c].astype(str).map(codes[c]).fillna(0).astype(int).values for c in CATS], 1)
    num = (d[NUMS] - mu) / sd
    x_num = np.concatenate([num.fillna(0).values, d[NUMS].isna().values.astype(float)], 1)  # value + "missing" flag
    return (torch.tensor(x_cat), torch.tensor(x_num, dtype=torch.float32),
            torch.tensor(d[TARGET].values, dtype=torch.float32))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(len(codes[c]) + 1, min(16, (len(codes[c]) + 1) // 2 + 1)) for c in CATS])
        width = sum(e.embedding_dim for e in self.emb) + 2 * len(NUMS)
        self.mlp = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, xc, xn):
        return self.mlp(torch.cat([e(xc[:, i]) for i, e in enumerate(self.emb)] + [xn], 1)).squeeze(1)


tr, va, te = tensors(train), tensors(val), tensors(test)
net, loss_fn = Net(), nn.L1Loss()
opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-5)
best, best_state, patience, history = 1e9, None, 0, []
t = time.time()
for epoch in range(40):
    net.train()
    for idx in torch.randperm(len(tr[2])).split(4096):
        opt.zero_grad()
        loss_fn(net(tr[0][idx], tr[1][idx]), tr[2][idx]).backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        v = loss_fn(net(va[0], va[1]), va[2]).item()
    history.append(v)
    if v < best - 1e-4:
        best, best_state, patience = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
    elif (patience := patience + 1) >= 5:
        break
net.load_state_dict(best_state); net.eval()
with torch.no_grad():
    nn_pred = net(te[0], te[1]).numpy()
results["Neural network (embeddings + MLP)"] = mae(nn_pred)
print(f"NN: {results['Neural network (embeddings + MLP)']:.3f} min, {len(history)} epochs, {time.time() - t:.0f}s")
plt.figure(figsize=(6, 3)); plt.plot(history, marker="o"); plt.title("NN validation MAE per epoch")
plt.xlabel("epoch"); plt.ylabel("MAE (min)"); save("08_nn_training")

# %% [markdown]
# ## 8. Results

# %%
res = pd.Series(results).sort_values()
print(res.round(3).to_string())
mlkit.save_metrics(OUT, {k: {"mae_min": v} for k, v in results.items()})

# %% [markdown]
# ## 8b. MLOps: track this session in MLflow, register the best model as `transit_delay_lgbm`
# (promoted to `champion` only if it beats the current one on the same test rows)

# %%
mlflow, prefix = mlkit.mlflow_setup("transit_delay")
best_lgb = fit_lgb(no_city)
with mlflow.start_run(run_name=f"train {pd.Timestamp.now():%Y-%m-%d %H:%M}"):
    mlkit.log_session(mlflow, {k: {"mae_min": v} for k, v in results.items()}, {
        "train_days": f"{train.date.min()} to {train.date.max()}",
        "test_days": f"{test.date.min()} to {test.date.max()}",
        "features": ",".join(no_city), "lgb_trees": best_lgb.best_iteration_})
    info = mlflow.lightgbm.log_model(best_lgb.booster_, "model", input_example=test[no_city].head(5))
mlkit.promote_if_better(mlflow, prefix + "transit_delay_lgbm", info.model_uri, test[no_city], test[TARGET])
colors = ["#5b3df5" if ("LightGBM" in k or "Neural" in k) else "#bbbbbb" for k in res.index]
res.plot.barh(color=colors, title="Test MAE, minutes (lower is better)"); save("09_results")

# %% [markdown]
# ## 9. Where is the model wrong?

# %%
test = test.assign(pred=gbm.predict(test[FEATURES]))
test["abs_err"] = (test["pred"] - test[TARGET]).abs()
fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
test.groupby("hod")["abs_err"].mean().plot(ax=ax[0], marker="o", title="LightGBM error by hour of day")
test.groupby("mode", observed=True)["abs_err"].mean().plot.bar(ax=ax[1], color="#d97757", title="Error by mode")
save("10_error_breakdown")
worst = (test.groupby("line_id", observed=True)["abs_err"].agg(["mean", "size"])
         .query("size >= 200").sort_values("mean", ascending=False).head(10).round(2))
print("Hardest lines to predict:\n", worst.to_string())
if test["disrupted"].any():
    print("\nMAE on disrupted test days:", round(test.loc[test.disrupted, "abs_err"].mean(), 3))

# %% [markdown]
# ## 10. What we found (run of 27 Oct 2026, data 10 to 26 Oct)
#
# **Data.** 835,182 stop-line-hours, 1,889 stops, 89 lines, no missing values. 17 full days; one disrupted day
# (Mon 12 Oct: median excess wait 4.4 min against a normal 0.34 min) kept out of training. 354 glitch values clipped (0.04%).
#
# **Correlation.** With the daily rhythm removed, the clearest link is **vehicles on the road: r = −0.29**: when fewer
# vehicles run than usual for that hour, people wait longer. NO₂ (+0.19) and grid load (+0.17) move weakly with the wait,
# most likely because all three follow traffic and activity, not because one causes the other. PM2.5 shows nothing (0.01).
#
# **Prediction (test = Fri 23 to Mon 26 Oct, never seen in training).**
#
# | Model | MAE (min) |
# |---|---|
# | Same as one hour ago | 1.374 |
# | Always 0 (trust the timetable) | 1.277 |
# | Average per stop/line/hour of day | 1.146 |
# | Neural network (embeddings + MLP) | 1.053 |
# | LightGBM | 1.027 |
# | **LightGBM without air/grid/city features** | **1.018** |
#
# - LightGBM beats the best simple rule by **11%** (1.146 → 1.018 min). The neural network gets most of the way (1.053)
#   but trees still win on a table this size, the usual result for tabular data.
# - **Air quality and grid load do not help the prediction**: removing them makes the model slightly *better*. The weak
#   correlation above is already explained by the hour, the stop and the recent delays.
# - "Same as one hour ago" is the worst rule: delays at one stop are noisy hour to hour, so the history of the same
#   hour on earlier days matters more than the last hour.
# - Hardest to predict: night buses (N04 to N16) and lines 72, 39, 41: few vehicles per hour, so one late bus moves the
#   average a lot.
#
# **Next.** More weeks of data (weekly patterns, a second disrupted day to test on), weather as a feature, and
# a second target: correcting STIB's own countdown.
