#!/usr/bin/env bash
# bash run.sh <task>     e.g. bash run.sh grid_load   (tasks: ls tasks/)
# 1. tasks/<task>/get_data.py   -> data/<task>/
# 2. tasks/<task>/notebook.py   -> outputs/<task>/notebook.ipynb (executed) + report.html + figures/ + metrics.csv
set -euo pipefail
cd "$(dirname "$0")"
task=${1:?usage: bash run.sh <task>   (one of: $(ls tasks | tr '\n' ' '))}
out=outputs/$task
mkdir -p "$out"
uv run python "tasks/$task/get_data.py"
uv run jupytext --to ipynb "tasks/$task/notebook.py" -o "$out/notebook.ipynb"
uv run jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$out/notebook.ipynb"
uv run jupyter nbconvert --to html "$out/notebook.ipynb" --output report
echo "done: $out/report.html"
