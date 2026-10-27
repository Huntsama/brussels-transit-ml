import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlkit import mae, mape, past_mean, time_split  # noqa: E402


def test_time_split_never_mixes_time():
    df = pd.DataFrame({"t": pd.date_range("2026-01-01", periods=10, freq="D")})
    train, val, test = time_split(df, "t", pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-08"))
    assert len(train) + len(val) + len(test) == len(df)
    assert train.t.max() < val.t.min() and val.t.max() < test.t.min()
    by_index = time_split(df.set_index("t"), df.set_index("t").index.to_series(),
                          pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-08"))
    assert [len(x) for x in by_index] == [len(train), len(val), len(test)]


def test_past_mean_uses_only_earlier_rows():
    df = pd.DataFrame({"k": ["a", "a", "b", "a", "b"], "v": [1.0, 3.0, 10.0, np.nan, 20.0]})
    got = past_mean(df["v"], df["k"])
    # first row of each group has no history; a NaN value is skipped, not counted as 0
    assert np.isnan(got[0]) and np.isnan(got[2])
    assert got[1] == 1.0 and got[3] == 2.0 and got[4] == 10.0


def test_metrics():
    assert mae([1, 2, 3], [1, 2, 5]) == 2 / 3
    assert round(mape([100, 200], [110, 180]), 6) == 10.0
