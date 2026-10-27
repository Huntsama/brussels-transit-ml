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
