"""The few helpers every task notebook shares: where files go, time splits, leak-free history, metrics, figures.

Kept to one file on purpose, so Colab only needs this file uploaded next to the data.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IN_COLAB = "google.colab" in sys.modules


def repo_root():
    here = Path.cwd()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() and (p / "tasks").exists():
            return p
    return here  # Colab / anywhere else: everything next to the notebook


def paths(task):
    """data dir (inputs) and output dir (figures, metrics) for a task; created if missing."""
    root = repo_root()
    data = root / "data" / task if (root / "tasks").exists() else root / "data"
    out = root / "outputs" / task if (root / "tasks").exists() else root / "outputs"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data.mkdir(parents=True, exist_ok=True)
    return data, out


def colab_upload(data_dir, names):
    """In Colab, ask for any missing input files; elsewhere do nothing."""
    missing = [n for n in names if not (Path(data_dir) / n).exists()]
    if IN_COLAB and missing:
        from google.colab import files
        print("Upload:", ", ".join(missing))
        for name, content in files.upload().items():
            (Path(data_dir) / name).write_bytes(content)


def time_split(df, col, val_start, test_start):
    """Split by time, never at random: train < val_start <= val < test_start <= test.
    col: a column name, or a Series of times aligned with df (e.g. df.index.to_series())."""
    t = df[col] if isinstance(col, str) else col
    return df[t < val_start], df[(t >= val_start) & (t < test_start)], df[t >= test_start]


def past_mean(values, keys):
    """Mean of earlier rows only, per group (rows must already be in time order). NaN in values = not counted."""
    v, seen = values.fillna(0), values.notna().astype(int)
    total = v.groupby(keys).cumsum() - v
    count = seen.groupby(keys).cumsum() - seen
    return (total / count).where(count > 0)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs(y_true - np.asarray(y_pred)) / np.abs(y_true)) * 100)


def save_fig(out_dir, name):
    import matplotlib.pyplot as plt
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "figures" / f"{name}.png", dpi=110)
    plt.show()


def save_metrics(out_dir, results):
    """results: {model name: {metric: value}} -> metrics.csv, printed sorted by the first metric."""
    df = pd.DataFrame(results).T
    df = df.sort_values(df.columns[0])
    df.round(4).to_csv(Path(out_dir) / "metrics.csv")
    print(df.round(3).to_string())
    return df


# --- MLOps: MLflow tracking + model registry -------------------------------------------------------------------
# On Databricks: the workspace's MLflow and Unity Catalog's registry (models live in brussels.ml.<name>).
# Anywhere else: a local MLflow database in outputs/ (see it with: uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db)

def mlflow_setup(task):
    """Point MLflow at the right place and select the task's experiment. Returns (mlflow, registry name prefix)."""
    import mlflow
    if "DATABRICKS_RUNTIME_VERSION" in os.environ:
        mlflow.set_registry_uri("databricks-uc")
        mlflow.set_experiment(f"/Shared/brussels/{task}")
        return mlflow, "brussels.ml."
    out = repo_root() / "outputs"
    mlflow.set_tracking_uri(f"sqlite:///{out / 'mlflow.db'}")  # the registry needs a database, not plain files
    if mlflow.get_experiment_by_name(task) is None:
        mlflow.create_experiment(task, artifact_location=(out / "mlruns").as_uri())
    mlflow.set_experiment(task)
    return mlflow, ""


def log_session(mlflow, results, params, figures_dir=None):
    """One parent run per training session, one child run per model, so the MLflow UI can compare them side by side."""
    import re
    for name, metrics in results.items():
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_param("model", name)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
    mlflow.log_params(params)
    best = min(results, key=lambda k: list(results[k].values())[0])
    mlflow.set_tag("best_model", best)
    mlflow.log_metrics({re.sub(r"[^\w]+", "_", f"best_{k}"): float(v) for k, v in results[best].items()})
    if figures_dir:
        mlflow.log_artifacts(str(figures_dir), "figures")


def promote_if_better(mlflow, name, model_uri, X, y, metric=mae):
    """Register the new model; give it the `champion` alias only if it beats the current champion on the SAME data.
    The forecast job always loads models:/<name>@champion, so a worse model can never go live by accident."""
    client = mlflow.MlflowClient()
    new = metric(y, mlflow.pyfunc.load_model(model_uri).predict(X))
    try:
        old = metric(y, mlflow.pyfunc.load_model(f"models:/{name}@champion").predict(X))
    except Exception as e:  # first run, or the features changed so the old model can't score this data
        print(f"no comparable champion ({type(e).__name__}): the new model becomes champion")
        old = None
    version = mlflow.register_model(model_uri, name).version
    alias = "champion" if old is None or new < old else "challenger"
    client.set_registered_model_alias(name, alias, version)
    print(f"{name} v{version}: {new:.3f} vs champion {old if old is None else round(old, 3)} -> {alias}")
    return version, alias
