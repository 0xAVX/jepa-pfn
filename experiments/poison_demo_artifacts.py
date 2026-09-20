"""Poison-demo artifacts (phoneme): PCA-2D, corruption flags, selections for
random/entropy/jepa-uniform-kc/jepa-guided-kc/guided-mix at 10% budget,
downstream AUCs (from figs/mechanism.csv protocol, recomputed for consistency).
Saves figs/poison_demo.npz. GPU ~10 min.
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
from src.ev import tabpfn_predict_proba

SEED = 0


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    X, y = openml_binary("phoneme")
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), 3404, replace=False)
    Xpool, ypool = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
    rng = np.random.RandomState(7)
    corr = np.zeros(len(ypool), bool)
    corr[rng.choice(len(ypool), int(0.10 * len(ypool)), replace=False)] = True
    y_dirty = ypool.copy()
    y_dirty[corr] = 1 - y_dirty[corr]
    k = 340

    u = oof_uncertainty(Xpool, y_dirty, seed=SEED)
    ent = u["entropy"].values
    est = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED)
    est.feats_ = Xpool.columns.tolist()
    sens = est._sensitivity(Xpool, y_dirty, n=500)
    fp = 0.1 + 0.8 * sens / (sens.max() + 1e-12)
    w = 1 + 2.0 * (ent / ent.max())
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, row_weights=w, feat_probs=fp,
                          seed=SEED)
    net_u, dev_u = train_plug(Xp, epochs=5, d_lat=32, seed=SEED)
    Z, Zu = embed(net, dev, Xp), embed(net_u, dev_u, Xp)
    top = np.argsort(-ent)[:2000]
    sels = {
        "random": np.random.RandomState(1).choice(len(ypool), k, replace=False),
        "entropy": np.argsort(-ent)[:k],
        "jepa-uniform": kcenter(Zu, k),
        "jepa-guided": kcenter(Z, k),
        "guided-mix": top[kcenter(Z[top], k)],
    }
    xy = PCA(n_components=2, random_state=0).fit_transform(
        (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)).astype(np.float32)
    out = {"xy": xy, "y": ypool.astype(np.int64), "corr": corr}
    for sname, sel in sels.items():
        p = tabpfn_predict_proba(Xpool.iloc[sel], y_dirty[sel], Xval, seed=SEED)
        a = roc_auc_score(yval, p)
        out[f"sel_{sname}"] = sel.astype(np.int64)
        out[f"auc_{sname}"] = np.float64(a)
        out[f"corrfrac_{sname}"] = float(corr[sel].mean())
        print(f"{sname}: corr={corr[sel].mean():.3f} AUC={a:.4f}", flush=True)
    np.savez_compressed("figs/poison_demo.npz", **out)
    print(f"saved figs/poison_demo.npz ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
