"""Load the scores already in outputs/<task>/metrics.csv into a local MLflow, to see the MLflow UI without training.

    uv run python demo/mlflow_from_results.py
    uv run mlflow ui --backend-store-uri sqlite:///outputs/mlflow.db     -> http://127.0.0.1:5000

The real runs (with model files and the registry) come from the notebooks on Databricks; these are tagged "imported".
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import mlkit  # noqa: E402

for task in ["grid_load", "transit_delay"]:
    csv = ROOT / "outputs" / task / "metrics.csv"
    if not csv.exists():
        continue
    results = pd.read_csv(csv, index_col=0).to_dict(orient="index")
    mlflow, _ = mlkit.mlflow_setup(task)
    with mlflow.start_run(run_name="imported from metrics.csv"):
        mlflow.set_tag("source", "imported, not retrained")
        mlkit.log_session(mlflow, results, {"note": "scores of the 27 Oct 2026 run"}, ROOT / "outputs" / task / "figures")
    print(f"{task}: {len(results)} models logged")
