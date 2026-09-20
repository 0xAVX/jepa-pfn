"""Precompute deterministic demo artifacts: PCA-2D pool, selections per
strategy x budget, TabPFN val scores. Demo loads this; nothing retrains live.
Datasets: phoneme, s6e9, NATICUSdroid. Saves figs/demo.npz. GPU ~30 min.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/dead/pfn-jepa/src")
sys.path.insert(0, "/home/dead/pfn-jepa/experiments")
sys.path.insert(0, "/home/dead/playground-series-s6e9")
from pfn_jepa.crossfit import oof_uncertainty
from pfn_jepa.estimator import PFNJEPAClassifier
from pfn_jepa.jepa import embed, prep, train_plug
from selection import kcenter
from run_matrix import openml_binary
from src.ev import load, tabpfn_predict_proba

SEED = 0
BUDGETS = [0.05, 0.1, 0.2, 0.4]
POOLS = {"phoneme": 3404, "s6e9": 8000, "NATICUSdroid": 5491}


def build(name, X, y):
    pool_n = POOLS[name]
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), min(pool_n, len(Xpool)),
                                             replace=False)
    Xpool, ypool = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
    print(f"{name}: pool={len(ypool)}", flush=True)

    u = oof_uncertainty(Xpool, ypool, seed=SEED)
    ent = u["entropy"].values
    est = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED)
    est.feats_ = Xpool.columns.tolist()
    sens = est._sensitivity(Xpool, ypool, n=500)
    fp = 0.1 + 0.8 * sens / (sens.max() + 1e-12)
    w = 1 + 2.0 * (ent / ent.max())
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, row_weights=w, feat_probs=fp,
                          seed=SEED)
    Z = embed(net, dev, Xp)
    Zr = Xp / (np.abs(Xp).max(axis=0) + 1e-9)
    xy = PCA(n_components=2, random_state=0).fit_transform(
        (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9))

    out = {"xy": xy.astype(np.float32), "y": ypool.astype(np.int64)}
    for b in BUDGETS:
        k = max(50, int(len(ypool) * b))
        top = np.argsort(-ent)[:max(k, min(2000, len(ypool)))]
        sels = {
            f"random": np.random.RandomState(1).choice(len(ypool), k, replace=False),
            f"entropy": np.argsort(-ent)[:k],
            f"raw-kcenter": kcenter(Zr, k),
            f"jepa-kcenter": kcenter(Z, k),
            f"guided-mix": top[kcenter(Z[top], k)],
        }
        for sname, sel in sels.items():
            p = tabpfn_predict_proba(Xpool.iloc[sel], ypool[sel], Xval, seed=SEED)
            a = roc_auc_score(yval, p)
            out[f"sel_{sname}_{b}"] = sel.astype(np.int64)
            out[f"auc_{sname}_{b}"] = np.float64(a)
            print(f"  {name} b={b} {sname}: {a:.4f}", flush=True)
    return out


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y, _, _, _ = load("/home/dead/playground-series-s6e9/data")
    datasets = {"s6e9": (X, y)}
    for nm in ["NATICUSdroid", "phoneme"]:
        datasets[nm] = openml_binary(nm)
    blob = {}
    for name, (X, y) in datasets.items():
        for k, v in build(name, X, y).items():
            blob[f"{name}/{k}"] = v
    np.savez_compressed("figs/demo.npz", **blob)
    print(f"saved figs/demo.npz ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
