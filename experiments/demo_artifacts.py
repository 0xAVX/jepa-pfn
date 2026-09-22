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

from pfn_jepa.data import kcenter, load_s6e9, openml_binary, tabpfn_predict_proba
from pfn_jepa.estimator import PFNJEPAClassifier
from pfn_jepa.jepa import embed, prep, train_plug
from tabpfn import TabPFNClassifier

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

    # seed protocol: 5% labeled; entropy from seed-fit on unlabeled rows
    sidx = []
    for k in (0, 1):
        kk = np.where(ypool == k)[0]
        sidx += np.random.RandomState(SEED).choice(
            kk, max(5, int(0.05 * len(ypool) * (ypool == k).mean())),
            replace=False).tolist()
    sidx = np.array(sidx)
    unl = np.array([i for i in range(len(ypool)) if i not in set(sidx)])
    n_seed = len(sidx)
    seed_clf = TabPFNClassifier(random_state=SEED)
    seed_clf.fit(Xpool.iloc[sidx], ypool[sidx])
    p_all = np.clip(seed_clf.predict_proba(Xpool)[:, 1], 1e-6, 1 - 1e-6)
    ent_all = -(p_all * np.log(p_all) + (1 - p_all) * np.log(1 - p_all))
    ent = ent_all[unl]
    est = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED)
    est.feats_ = Xpool.columns.tolist()
    sens = est._sensitivity(Xpool.iloc[sidx], ypool[sidx],
                            n=min(500, len(sidx)))
    fp = 0.1 + 0.8 * sens / (sens.max() + 1e-12)
    w = 1 + 2.0 * (ent_all / ent_all.max())
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, row_weights=w, feat_probs=fp,
                          seed=SEED)
    Z = embed(net, dev, Xp)[unl]
    Zr = (Xp / (np.abs(Xp).max(axis=0) + 1e-9))[unl]
    xy = PCA(n_components=2, random_state=0).fit_transform(
        (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9))

    out = {"xy": xy.astype(np.float32), "y": ypool.astype(np.int64)}
    pos = np.argsort(-ent)
    take = lambda s: unl[s]
    for b in BUDGETS:
        k = max(n_seed + 25, int(len(ypool) * b))
        need = k - n_seed
        top = pos[:max(need, min(2000, len(unl)))]
        sels = {
            f"random": np.concatenate(
                [sidx, take(np.random.RandomState(1).choice(len(unl), need,
                                                            replace=False))]),
            f"entropy": np.concatenate([sidx, take(pos[:need])]),
            f"raw-kcenter": np.concatenate([sidx, take(kcenter(Zr, need))]),
            f"jepa-kcenter": np.concatenate([sidx, take(kcenter(Z, need))]),
            f"guided-mix": np.concatenate([sidx, take(top[kcenter(Z[top], need)])]),
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
    try:
        X, y, _, _, _ = load_s6e9("data")
        datasets = {"s6e9": (X, y)}
    except Exception as e:
        print("skip s6e9:", e, flush=True)
        datasets = {}
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
