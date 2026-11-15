"""Step 2: fit the models on the standard split (same as the notebook), log to MLflow, promote the best LightGBM.

    uv run python tasks/transit_delay/train.py [--nn] [--no-mlflow]

Reads data/transit_delay/features.parquet (from features.py) and config.yaml. Writes outputs/transit_delay/:
  models/lgbm.txt, models/lgbm_no_city.txt   LightGBM boosters
  test_predictions.parquet                   every model's prediction for every test row (evaluate.py scores them)
  train_summary.json                         split, trees, timings, validation MAE, data snapshot id, git commit, NN curve
"""
import argparse
import json
import subprocess
import time

import pandas as pd

from common import (BASELINES, DATA, OUT, ROOT, baselines, fit_lgb, fit_nn, frames, full_days, load_config,
                    load_features, mae, mlkit, standard_days)

LGB_MODELS = {"lgbm": "LightGBM", "lgbm_no_city": "LightGBM without air/grid/city features"}
NN_NAME = "Neural network (embeddings + MLP)"


def git_commit():
    def git(*a):
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    code_changed = git("status", "--porcelain", "--", "src", "tasks", "tests", "run.sh", "pyproject.toml")
    return git("rev-parse", "--short", "HEAD") + ("-dirty" if code_changed else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nn", action="store_true", help="also train the neural network")
    ap.add_argument("--no-mlflow", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    target, feats = cfg["target"], cfg["features"]
    feature_sets = {"lgbm": feats, "lgbm_no_city": [f for f in feats if f not in cfg["city_features"]]}
    meta = json.loads((DATA / "features.meta.json").read_text())

    df = load_features(columns=["hour", "date", "disrupted", target] + feats)
    tr_d, va_d, te_d = standard_days(full_days(df), cfg["split"])
    train, val, test = frames(df, tr_d, va_d, te_d)
    del df
    fmt = lambda d: f"{d:%a %d %b}"  # noqa: E731
    print(f"train {len(train):,} rows ({fmt(tr_d[0])} -> {fmt(tr_d[-1])}), val {len(val):,} "
          f"({fmt(va_d[0])}, {fmt(va_d[-1])}), test {len(test):,} ({fmt(te_d[0])} -> {fmt(te_d[-1])})")

    preds = baselines(train, test, target)
    summary = {"snapshot_id": meta["snapshot_id"], "git_commit": git_commit(),
               "train_days": [str(tr_d[0].date()), str(tr_d[-1].date())],
               "val_days": [str(d.date()) for d in va_d], "test_days": [str(te_d[0].date()), str(te_d[-1].date())],
               "rows": {"train": len(train), "val": len(val), "test": len(test)}, "models": {}}
    # the NN runs before LightGBM: on the desktop (i5-2400, WSL) torch's Adam step died with "Illegal instruction"
    # twice when it ran right after the two long LightGBM fits in the same process, and not when it ran first
    if args.nn:
        t = time.time()
        predict, history = fit_nn(train, val, cfg)
        preds[NN_NAME] = predict(test)
        summary["models"][NN_NAME] = {"epochs": len(history), "fit_seconds": round(time.time() - t),
                                      "val_mae_per_epoch": history}
        print(f"NN: test {mae(preds[NN_NAME], test[target]):.3f} min, {len(history)} epochs, {time.time() - t:.0f}s")
    (OUT / "models").mkdir(exist_ok=True)
    boosters = {}
    for key, name in LGB_MODELS.items():
        m = fit_lgb(train, val, feature_sets[key], cfg)
        preds[name] = m.predict(test[feature_sets[key]])
        m.booster_.save_model(str(OUT / "models" / f"{key}.txt"), num_iteration=m.best_iteration_)
        boosters[key] = m
        summary["models"][name] = {"trees": int(m.best_iteration_), "fit_seconds": round(m.fit_seconds),
                                   "val_mae": mae(m.predict(val[feature_sets[key]]), val[target]),
                                   "features": feature_sets[key]}
        print(f"{name}: test {mae(preds[name], test[target]):.3f} min, {m.best_iteration_} trees, {m.fit_seconds:.0f}s")

    results = {k: {"mae_min": mae(p, test[target])} for k, p in preds.items()}
    out = test[["hour", "stop_id", "line_id", "mode", "hod", "dow", "disrupted", target]].reset_index(drop=True)
    for k, p in preds.items():
        out[k] = pd.Series(p, dtype="float32")
    out.to_parquet(OUT / "test_predictions.parquet", index=False)

    # the LightGBM to register: the better one on the VALIDATION days (choosing on the test would leak)
    reg = min(LGB_MODELS, key=lambda k: summary["models"][LGB_MODELS[k]]["val_mae"])
    summary["registered"] = LGB_MODELS[reg]
    summary["baselines"] = BASELINES
    if not args.no_mlflow:
        mlflow, prefix = mlkit.mlflow_setup("transit_delay")
        with mlflow.start_run(run_name=f"train {pd.Timestamp.now():%Y-%m-%d %H:%M}"):
            mlkit.log_session(mlflow, results, {
                "train_days": " to ".join(summary["train_days"]), "test_days": " to ".join(summary["test_days"]),
                "features": ",".join(feature_sets[reg]), "lgb_trees": boosters[reg].best_iteration_,
                "data_snapshot": meta["snapshot_id"], "git_commit": summary["git_commit"]})
            mlflow.set_tags({"data_snapshot": meta["snapshot_id"], "git_commit": summary["git_commit"],
                             "data_first_hour_utc": meta["first_hour_utc"], "data_last_hour_utc": meta["last_hour_utc"]})
            mlflow.log_dict(meta, "data_snapshot.json")
            X = test[feature_sets[reg]]
            info = mlflow.lightgbm.log_model(boosters[reg].booster_, "model", input_example=X.head(5))
        version, alias = mlkit.promote_if_better(mlflow, prefix + cfg["registry_name"], info.model_uri, X, test[target])
        summary["registry"] = {"name": prefix + cfg["registry_name"], "version": int(version), "alias": alias}
    (OUT / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(pd.Series({k: v["mae_min"] for k, v in results.items()}).sort_values().round(3).to_string())


if __name__ == "__main__":
    main()
