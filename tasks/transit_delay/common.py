"""Shared by train.py, evaluate.py and backtest.py: config, loading the feature table, splits, baselines, models."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mlkit  # noqa: E402

DATA, OUT = mlkit.paths("transit_delay")


def load_config(path=HERE / "config.yaml"):
    import yaml
    return yaml.safe_load(Path(path).read_text())


def load_features(path=None, columns=None, categorical=("stop_id", "line_id", "mode")):
    """features.parquet -> DataFrame; strings become categories with sorted categories (as the notebook's astype)."""
    import pyarrow.parquet as pq
    path = path or DATA / "features.parquet"
    cats = [c for c in categorical if columns is None or c in columns]
    df = pq.read_table(path, columns=columns, read_dictionary=cats).to_pandas(date_as_object=False)
    for c in cats:
        df[c] = df[c].cat.set_categories(sorted(df[c].cat.categories))
    return df


def mae(pred, y):
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64))))


# --- splits ------------------------------------------------------------------------------------------------------

def full_days(df):
    """The sorted local dates in the table (features.parquet only holds full days) as Timestamps."""
    return [pd.Timestamp(d) for d in sorted(pd.unique(df["date"]))]


def standard_days(days, split):
    """Last quarter of the days = test, the val_days before = validation, the rest = training (notebook rule)."""
    days = sorted(days)
    n_test = max(split["min_test_days"], int(len(days) * split["test_fraction"]))
    nv = split["val_days"]
    return days[:-n_test - nv], days[-n_test - nv:-n_test], days[-n_test:]


def make_folds(days, n_folds, test_days, val_days, min_train_days):
    """Weekly folds, newest first: test = a block of test_days consecutive days ending on the last day (then one week
    earlier, ...), validation = the val_days before it, training = every earlier day. Folds with fewer than
    min_train_days training days are dropped. Returns [(train_days, val_days, test_days)]."""
    days = sorted(days)
    folds = []
    for k in range(n_folds):
        end = len(days) - k * test_days
        start = end - test_days
        if start - val_days < min_train_days:
            break
        folds.append((days[:start - val_days], days[start - val_days:start], days[start:end]))
    return folds


def frames(df, train_days, val_days, test_days):
    """Rows for each part. Disrupted days never train or validate; they stay in the test (reported as they are)."""
    d = df["date"]
    normal = ~df["disrupted"]
    return (df[d.isin(train_days) & normal], df[d.isin(val_days) & normal], df[d.isin(test_days)])


# --- models ------------------------------------------------------------------------------------------------------

BASELINES = ["Always 0 (trust the timetable)", "Average per stop/line/hour of day", "Same as one hour ago"]


def baselines(train, test, target):
    """The three simple rules of the notebook -> {name: predictions for test}."""
    hod_mean = train.groupby(["stop_id", "line_id", "hod"], observed=True)[target].mean().rename("p").reset_index()
    p = test[["stop_id", "line_id", "hod"]].merge(hod_mean, how="left", on=["stop_id", "line_id", "hod"])["p"]
    return {
        BASELINES[0]: np.zeros(len(test), dtype=np.float32),
        BASELINES[1]: p.fillna(train[target].median()).to_numpy(np.float32),
        BASELINES[2]: test["prev_hour_ewt"].fillna(0).to_numpy(np.float32),
    }


def fit_lgb(train, val, features, cfg):
    import lightgbm as lgb
    t = time.time()
    m = lgb.LGBMRegressor(**cfg["lightgbm"])
    m.fit(train[features], train[cfg["target"]], eval_set=[(val[features], val[cfg["target"]])],
          callbacks=[lgb.early_stopping(cfg["early_stopping_rounds"], verbose=False)])
    m.fit_seconds = time.time() - t
    return m


NN_CATS = ["stop_id", "line_id", "mode", "hod", "dow"]
NN_NUMS = ["scheduled_wait_min", "prev_hour_ewt", "prev_days_same_hod_ewt", "city_prev_ewt", "no2_prev", "load_prev"]


def fit_nn(train, val, cfg):
    """The notebook's network: an embedding per stop/line/mode/hour/weekday + numeric features (value + missing flag)
    -> MLP 128 -> 64 -> 1, L1 loss, early stopping on validation. Returns (predict function, val MAE per epoch)."""
    import torch
    from torch import nn
    c = cfg["nn"]
    torch.manual_seed(c["seed"]); np.random.seed(c["seed"])
    target = cfg["target"]
    vocab = {k: pd.Index(pd.unique(train[k])) for k in NN_CATS}  # index + 1; 0 = not seen in training
    mu = train[NN_NUMS].astype("float64").mean()
    sd = train[NN_NUMS].astype("float64").std().replace(0, 1)

    def tensors(d):
        x_cat = np.stack([vocab[k].get_indexer(d[k]) + 1 for k in NN_CATS], 1)
        num = (d[NN_NUMS].astype("float64") - mu) / sd
        x_num = np.concatenate([num.fillna(0).values, d[NN_NUMS].isna().values], 1).astype(np.float32)
        return torch.tensor(x_cat, dtype=torch.long), torch.tensor(x_num), torch.tensor(d[target].values, dtype=torch.float32)

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.ModuleList([nn.Embedding(len(vocab[k]) + 1, min(16, (len(vocab[k]) + 1) // 2 + 1))
                                      for k in NN_CATS])
            width = sum(e.embedding_dim for e in self.emb) + 2 * len(NN_NUMS)
            self.mlp = nn.Sequential(nn.Linear(width, 128), nn.ReLU(), nn.Dropout(0.1),
                                     nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

        def forward(self, xc, xn):
            return self.mlp(torch.cat([e(xc[:, i]) for i, e in enumerate(self.emb)] + [xn], 1)).squeeze(1)

    tr, va = tensors(train), tensors(val)
    net, loss_fn = Net(), nn.L1Loss()
    opt = torch.optim.AdamW(net.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    best, best_state, patience, history = 1e9, None, 0, []
    for _ in range(c["epochs"]):
        net.train()
        for idx in torch.randperm(len(tr[2])).split(c["batch"]):
            opt.zero_grad()
            loss_fn(net(tr[0][idx], tr[1][idx]), tr[2][idx]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = loss_fn(net(va[0], va[1]), va[2]).item()
        history.append(v)
        if v < best - 1e-4:
            best, best_state, patience = v, {k: x.clone() for k, x in net.state_dict().items()}, 0
        elif (patience := patience + 1) >= c["patience"]:
            break
    net.load_state_dict(best_state); net.eval()
    del tr, va

    def predict(d):
        xc, xn, _ = tensors(d)
        with torch.no_grad():
            return np.concatenate([net(xc[i:i + 65536], xn[i:i + 65536]).numpy() for i in range(0, len(xc), 65536)])
    return predict, history
