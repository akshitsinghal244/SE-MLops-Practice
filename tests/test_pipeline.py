from common import params
from evaluate import passes_gate


def test_params_sections():
    p = params()
    for key in ("data", "ml", "dl", "gate"):
        assert key in p


def test_gate():
    assert passes_gate({"ml": 0.95, "dl": 0.93}, 0.9)
    assert not passes_gate({"ml": 0.95, "dl": 0.5}, 0.9)


def test_ml_model_learns():
    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier

    X, y = load_digits(return_X_y=True)
    m = RandomForestClassifier(n_estimators=20, random_state=0).fit(X[:1000], y[:1000])
    assert m.score(X[1000:], y[1000:]) > 0.8


def test_dl_forward_shape():
    import torch
    from train_dl import build_model

    assert build_model(64, 32)(torch.zeros(2, 64)).shape == (2, 10)
