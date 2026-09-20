import numpy as np
import pandas as pd

from pfn_jepa.estimator import PFNJEPAClassifier


def toy(n=600, seed=0):
    rng = np.random.RandomState(seed)
    x1 = rng.randn(n)
    x2 = rng.randn(n)
    cat = rng.choice(["a", "b", "c"], n)
    y = ((x1 + 0.5 * x2 + (cat == "a")) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "c": cat}), y


def test_plain_loop():
    X, y = toy()
    m = PFNJEPAClassifier(d_lat=16, epochs=2, curriculum=False,
                          guided_masks=False, mc_samples=0, seed=0)
    m.fit(X.iloc[:400], y[:400])
    p = m.predict_proba(X.iloc[400:])
    assert p.shape == (200, 2)
    assert np.isfinite(p).all() and (p >= 0).all()
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(y[400:], p[:, 1]) > 0.7


def test_full_loop():
    X, y = toy()
    m = PFNJEPAClassifier(d_lat=16, epochs=2, mc_samples=2, seed=0)
    m.fit(X.iloc[:400], y[:400])
    p = m.predict_proba(X.iloc[400:410])
    assert p.shape == (10, 2)
    e = m.explain_uncertainty(X.iloc[[410]])
    assert set(e) == {"tabpfn_entropy", "jepa_surprise", "fragile_features",
                      "completion_disagreement"}
