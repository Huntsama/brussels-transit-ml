"""transit_delay backtest: folds never train or validate on the week they test."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tasks" / "transit_delay"))
from common import frames, make_folds  # noqa: E402


def test_backtest_folds_never_train_on_their_test_week():
    days = list(pd.date_range("2026-10-10", "2026-11-14", freq="D"))
    folds = make_folds(days, n_folds=4, test_days=7, val_days=2, min_train_days=14)
    assert len(folds) == 2  # 36 days: a third fold would have only 13 training days
    assert folds[0][2][-1] == days[-1]
    tested = set()
    for tr, va, te in folds:
        assert len(te) == 7 and len(va) == 2 and len(tr) >= 14
        assert max(tr) < min(va) and max(va) < min(te)
        assert not (set(te) & tested)  # test weeks do not overlap
        tested |= set(te)
    df = pd.DataFrame({"date": np.repeat(days, 3), "disrupted": np.repeat([d.day == 12 for d in days], 3), "y": 1.0})
    for tr, va, te in folds:
        train, val, test = frames(df, tr, va, te)
        assert not train["date"].isin(te).any() and not val["date"].isin(te).any()
        assert not train["disrupted"].any() and not val["disrupted"].any()
        assert len(test) == 7 * 3 and len(set(train.index) & set(test.index)) == 0
