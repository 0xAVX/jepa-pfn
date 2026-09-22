"""Self-contained data + inference helpers (no sibling repos, no absolute paths).

TabPFN chunked predict (small GPUs), stratified subsample, S6E9 loader
(download: kaggle competitions download -c playground-series-s6e9),
OpenML binary loader, greedy k-center.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def tabpfn_predict_proba(Xtr, ytr, Xte, chunk=10000, seed=0):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(random_state=seed)
    clf.fit(Xtr, ytr)
    return np.concatenate([clf.predict_proba(Xte[i:i + chunk])[:, 1]
                           for i in range(0, len(Xte), chunk)])


def stratified_subsample(X, y, n, seed=0):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    n_pos = int(round(n * np.mean(y)))
    take = np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(np.where(y == 0)[0], n - n_pos,
                                      replace=False)])
    rng.shuffle(take)
    return X.iloc[take], y[take]


CATS_S6E9 = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
             "Subsidy_Available", "Range_Anxiety_Level"]


def load_s6e9(data_dir="data", nrows=None):
    data_dir = Path(data_dir)
    tr = pd.read_csv(data_dir / "train.csv", nrows=nrows)
    te = pd.read_csv(data_dir / "test.csv", nrows=nrows)
    feats = [c for c in tr.columns if c not in ("id", "Will_Buy_EV")]
    for c in CATS_S6E9:
        tr[c] = tr[c].astype("category")
        te[c] = te[c].astype("category")
    y = (tr["Will_Buy_EV"] == "Yes").astype(int).values
    return tr[feats], y, te[feats], te["id"].values, feats


def openml_binary(name):
    from sklearn.datasets import fetch_openml
    from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
    d = fetch_openml(name=name, as_frame=True, parser="auto")
    X, y = d.data.copy(), d.target
    cat = [c for c in X.columns if str(X[c].dtype) in ("category", "object")]
    num = [c for c in X.columns if c not in cat]
    if cat:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat] = enc.fit_transform(X[cat].astype(str))
    X[num] = X[num].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(-1)
    y = LabelEncoder().fit_transform(pd.Series(y).astype(str).values)
    if y.mean() > 0.5:
        y = 1 - y
    return X.reset_index(drop=True), y


def kcenter(Z: np.ndarray, k: int, seed=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    n = len(Z)
    sel = [rng.choice(n)]
    dmin = np.linalg.norm(Z - Z[sel[0]], axis=1)
    for _ in range(1, k):
        i = int(np.argmax(dmin))
        sel.append(i)
        dmin = np.minimum(dmin, np.linalg.norm(Z - Z[i], axis=1))
    return np.array(sel)
