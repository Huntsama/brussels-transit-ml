"""Step 3: score every model's test predictions (from train.py) and draw the result figures.

    uv run python tasks/transit_delay/evaluate.py

Writes outputs/transit_delay/: metrics.csv (MAE per model, baselines included), hardest_lines.csv,
figures/07_importance.png, 08_nn_training.png (if the NN was trained), 09_results.png, 10_error_breakdown.png
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from common import OUT, load_config, mae, mlkit  # noqa: E402

plt.rcParams.update({"figure.figsize": (11, 4), "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def save(name):
    mlkit.save_fig(OUT, name)
    plt.close("all")


def main():
    import lightgbm as lgb
    cfg = load_config()
    target = cfg["target"]
    summary = json.loads((OUT / "train_summary.json").read_text())
    test = pd.read_parquet(OUT / "test_predictions.parquet")
    models = [c for c in test.columns if c in summary["baselines"] or c in summary["models"]]
    results = {m: {"mae_min": mae(test[m], test[target])} for m in models}
    res = mlkit.save_metrics(OUT, results)["mae_min"]

    booster = lgb.Booster(model_file=str(OUT / "models" / "lgbm.txt"))
    imp = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values()
    (imp / imp.sum()).plot.barh(color="#5b3df5", title="What LightGBM relies on (share of gain)"); save("07_importance")

    nn = summary["models"].get("Neural network (embeddings + MLP)")
    if nn:
        plt.figure(figsize=(6, 3)); plt.plot(nn["val_mae_per_epoch"], marker="o")
        plt.title("NN validation MAE per epoch"); plt.xlabel("epoch"); plt.ylabel("MAE (min)"); save("08_nn_training")

    colors = ["#5b3df5" if ("LightGBM" in k or "Neural" in k) else "#bbbbbb" for k in res.index]
    res.sort_values().plot.barh(color=colors, title="Test MAE, minutes (lower is better)"); save("09_results")

    test["abs_err"] = (test["LightGBM"].astype("float64") - test[target]).abs()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.5))
    test.groupby("hod")["abs_err"].mean().plot(ax=ax[0], marker="o", title="LightGBM error by hour of day")
    test.groupby("mode", observed=True)["abs_err"].mean().plot.bar(ax=ax[1], color="#d97757", title="Error by mode")
    save("10_error_breakdown")
    worst = (test.groupby("line_id", observed=True)["abs_err"].agg(["mean", "size"])
             .query("size >= 200").sort_values("mean", ascending=False).head(10).round(3))
    worst.to_csv(OUT / "hardest_lines.csv")
    print("Hardest lines to predict:\n", worst.to_string())
    if test["disrupted"].any():
        print("MAE on disrupted test days:", round(test.loc[test.disrupted, "abs_err"].mean(), 3))


if __name__ == "__main__":
    main()
