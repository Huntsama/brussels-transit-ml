#!/usr/bin/env bash
# bash run.sh <task>     e.g. bash run.sh grid_load   (tasks: ls tasks/)
# 1. tasks/<task>/get_data.py   -> data/<task>/          (skip with FETCH=0, e.g. when the data was copied in by hand)
# 2. if the task has pipeline steps (transit_delay: features.py, train.py, evaluate.py, backtest.py), run them
# 3. tasks/<task>/notebook.py   -> outputs/<task>/notebook.ipynb (executed) + report.html + figures/ + metrics.csv
#    (for transit_delay the notebook is a read-only report over the files the steps wrote)
# 4. tasks/<task>/fair_comparison.py, if the task has one (grid_load: weather forecasts instead of measured)
set -euo pipefail
cd "$(dirname "$0")"
task=${1:?usage: bash run.sh <task>   (one of: $(ls tasks | tr '\n' ' '))}
out=outputs/$task
mkdir -p "$out"
if [ "${FETCH:-1}" != 0 ]; then uv run python "tasks/$task/get_data.py"; fi
if [ -f "tasks/$task/features.py" ]; then
  uv run python "tasks/$task/features.py"
  uv run python "tasks/$task/train.py" --nn
  uv run python "tasks/$task/evaluate.py"
  uv run python "tasks/$task/backtest.py"
fi
uv run jupytext --to ipynb "tasks/$task/notebook.py" -o "$out/notebook.ipynb"
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$out/notebook.ipynb"
uv run jupyter nbconvert --to html "$out/notebook.ipynb" --output report
if [ -f "tasks/$task/fair_comparison.py" ]; then uv run python "tasks/$task/fair_comparison.py"; fi
echo "done: $out/report.html"
