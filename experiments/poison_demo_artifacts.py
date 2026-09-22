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

from pfn_jepa.data import kcenter, openml_binary, tabpfn_predict_proba
from pfn_jepa.estimator import PFNJEPAClassifier
from pfn_jepa.jepa import embed, prep, train_plug
from tabpfn import TabPFNClassifier

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
    # seed protocol on dirty labels: seed labeled, rest ranked unlabeled
    sidx = []
    for k in (0, 1):
        kk = np.where(y_dirty == k)[0]
        sidx += np.random.RandomState(SEED).choice(
            kk, max(5, int(0.05 * len(ypool) * (y_dirty == k).mean())),
            replace=False).tolist()
    sidx = np.array(sidx)
    unl = np.array([i for i in range(len(ypool)) if i not in set(sidx)])
    n_seed = len(sidx)
    k = 340
    need = k - n_seed

    seed_clf = TabPFNClassifier(random_state=SEED)
    seed_clf.fit(Xpool.iloc[sidx], y_dirty[sidx])
    p_all = np.clip(seed_clf.predict_proba(Xpool)[:, 1], 1e-6, 1 - 1e-6)
    ent_all = -(p_all * np.log(p_all) + (1 - p_all) * np.log(1 - p_all))
    ent = ent_all[unl]
    est = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED)
    est.feats_ = Xpool.columns.tolist()
    sens = est._sensitivity(Xpool.iloc[sidx], y_dirty[sidx],
                            n=min(500, len(sidx)))
    fp = 0.1 + 0.8 * sens / (sens.max() + 1e-12)
    w = 1 + 2.0 * (ent_all / ent_all.max())
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, row_weights=w, feat_probs=fp,
                          seed=SEED)
    net_u, dev_u = train_plug(Xp, epochs=5, d_lat=32, seed=SEED)
    Z, Zu = embed(net, dev, Xp)[unl], embed(net_u, dev_u, Xp)[unl]
    pos = np.argsort(-ent)
    top = pos[:max(need, min(2000, len(unl)))]
    take = lambda s: np.concatenate([sidx, unl[s]])
    sels = {
        "random": take(np.random.RandomState(1).choice(len(unl), need,
                                                       replace=False)),
        "entropy": take(pos[:need]),
        "jepa-uniform": take(kcenter(Zu, need)),
        "jepa-guided": take(kcenter(Z, need)),
        "guided-mix": take(top[kcenter(Z[top], need)]),
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
