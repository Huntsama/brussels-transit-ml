"""The promotion gate: a new model becomes champion only if it beats the current champion on the same data."""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mlkit import promote_if_better  # noqa: E402


def fake_mlflow(new_pred, champion_pred):
    """Just enough of MLflow's API for the gate; champion_pred=None means no champion yet."""
    aliases = {}

    class Model:
        def __init__(self, pred): self.pred = pred
        def predict(self, X): return self.pred

    def load_model(uri):
        if uri.endswith("@champion"):
            if champion_pred is None:
                raise LookupError("no champion")
            return Model(champion_pred)
        return Model(new_pred)

    class Client:
        def set_registered_model_alias(self, name, alias, version): aliases[alias] = version

    m = types.SimpleNamespace(pyfunc=types.SimpleNamespace(load_model=load_model), MlflowClient=Client,
                              register_model=lambda uri, name: types.SimpleNamespace(version=7))
    return m, aliases


y = np.array([10.0, 20.0, 30.0])


def test_first_model_becomes_champion():
    m, aliases = fake_mlflow(y + 1, None)
    assert promote_if_better(m, "x", "runs:/1/model", None, y) == (7, "champion")


def test_better_model_is_promoted():
    m, aliases = fake_mlflow(y + 1, y + 2)
    assert promote_if_better(m, "x", "runs:/1/model", None, y)[1] == "champion"


def test_worse_model_stays_challenger():
    m, aliases = fake_mlflow(y + 3, y + 2)
    assert promote_if_better(m, "x", "runs:/1/model", None, y)[1] == "challenger"
    assert "champion" not in aliases  # the live model was not replaced
