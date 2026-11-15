"""Step 4: rolling backtest. One standard split can be lucky; here every recent week is a test in turn.

    uv run python tasks/transit_delay/backtest.py [--nn]

Folds (config.yaml `backtest`): test = 7 consecutive days ending on the last full day, then the week before, ...;
validation (early stopping) = the 2 days before the test week; training = every earlier day, disrupted days out.
A fold needs at least `min_train_days` training days. Writes outputs/transit_delay/backtest.csv and backtest.md.
"""
import argparse
import gc

import pandas as pd

from common import BASELINES, OUT, baselines, fit_lgb, fit_nn, frames, full_days, load_config, load_features, mae, make_folds

NAMES = {"lgbm": "LightGBM", "lgbm_no_city": "LightGBM without air/grid/city features",
         "nn": "Neural network (embeddings + MLP)"}


def report(bt, folds_info):
    fmt = lambda x: f"{x:.3f}"  # noqa: E731
    wide = bt.pivot(index="model", columns="fold", values="mae")
    wide["mean"], wide["std"] = wide.mean(axis=1), wide.std(axis=1)
    wide = wide.sort_values("mean")
    best_base = bt[bt.model.isin(BASELINES)].groupby("fold")["mae"].min()
    best_base_name = bt[bt.model.isin(BASELINES)].loc[lambda d: d.groupby("fold")["mae"].idxmin(), ["fold", "model"]]
    lgbm = bt[bt.model == "LightGBM"].set_index("fold")["mae"]
    gain = (1 - lgbm / best_base) * 100

    lines = ["# transit_delay: rolling backtest", "",
             "Each fold trains on every day before it (disrupted days out), stops early on the 2 days before the "
             "test week, and is scored on the week (MAE, minutes; lower is better).", "",
             "| Fold | Train days | Validation | Test week | Test rows |", "|---|---|---|---|---|"]
    for f in folds_info:
        lines.append(f"| {f['fold']} | {f['train']} ({f['n_train_days']} days) | {f['val']} | {f['test']} | {f['test_rows']:,} |")
    lines += ["", "| Model | " + " | ".join(f"fold {c}" for c in wide.columns[:-2]) + " | mean ± std |",
              "|---|" + "---|" * (len(wide.columns) - 1)]
    for model, r in wide.iterrows():
        cells = [fmt(r[c]) for c in wide.columns[:-2]]
        lines.append(f"| {model} | " + " | ".join(cells) + f" | {r['mean']:.3f} ± {r['std']:.3f} |")
    lines += ["", "LightGBM vs the best simple rule of each fold:", ""]
    for fold, g in gain.items():
        name = best_base_name.set_index("fold").loc[fold, "model"]
        lines.append(f"- fold {fold}: {lgbm[fold]:.3f} vs {best_base[fold]:.3f} ({name}): **{g:.1f}% better**")
    lines += ["", f"Mean gain {gain.mean():.1f}% (min {gain.min():.1f}%, max {gain.max():.1f}%) over "
              f"{len(gain)} fold{'s' if len(gain) > 1 else ''}.", ""]
    always = all(lgbm[f] < best_base[f] for f in gain.index)
    lines += ["**Reading it.** " + (
        f"LightGBM beat every simple rule in all {len(gain)} folds" if always else
        "LightGBM did NOT beat the best simple rule in every fold") +
        f"; its edge ranges from {gain.min():.1f}% to {gain.max():.1f}%. "
        f"With {len(gain)} fold{'s' if len(gain) > 1 else ''}, the ± is a rough spread, not a confidence interval. "
        "Later folds train on more days, so part of a difference between folds is more history, part is the week "
        "itself (holidays, disruptions). The disrupted-day flag is decided once on all the data (median rule), so it "
        "is not a forecast of disruptions; disrupted days stay in the test weeks and are scored like any other day.", ""]
    return "\n".join(lines), gain


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    target, feats = cfg["target"], cfg["features"]
    sets = {"lgbm": feats, "lgbm_no_city": [f for f in feats if f not in cfg["city_features"]]}
    b = cfg["backtest"]
    df = load_features(columns=["date", "disrupted", target] + feats)
    folds = make_folds(full_days(df), b["n_folds"], b["test_days"], b["val_days"], b["min_train_days"])
    print(f"{len(folds)} folds (newest first)")
    rows, info = [], []
    d = lambda x: f"{x:%a %d %b}"  # noqa: E731
    for i, (tr_d, va_d, te_d) in enumerate(folds, 1):
        train, val, test = frames(df, tr_d, va_d, te_d)
        info.append({"fold": i, "train": f"{d(tr_d[0])} -> {d(tr_d[-1])}", "n_train_days": len(tr_d),
                     "val": f"{d(va_d[0])}, {d(va_d[-1])}", "test": f"{d(te_d[0])} -> {d(te_d[-1])}",
                     "test_rows": len(test)})
        print(info[-1])
        preds = baselines(train, test, target)
        for key, cols in sets.items():
            m = fit_lgb(train, val, cols, cfg)
            preds[NAMES[key]] = m.predict(test[cols])
            print(f"  {NAMES[key]}: {m.best_iteration_} trees, {m.fit_seconds:.0f}s")
            del m
        if args.nn:
            predict, _ = fit_nn(train, val, cfg)
            preds[NAMES["nn"]] = predict(test)
        for name, p in preds.items():
            rows.append({"fold": i, "test_start": str(te_d[0].date()), "test_end": str(te_d[-1].date()),
                         "train_days": len(tr_d), "model": name, "mae": round(mae(p, test[target]), 4)})
            print(f"  {rows[-1]['mae']:.3f}  {name}")
        del train, val, test, preds
        gc.collect()
    bt = pd.DataFrame(rows)
    bt.to_csv(OUT / "backtest.csv", index=False)
    md, gain = report(bt, info)
    (OUT / "backtest.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
