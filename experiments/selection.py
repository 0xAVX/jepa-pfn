"""Selection baselines: label-budget curves on guided JEPA representations.

Question: does the guided representation add anything over TabPFN uncertainty
alone? Strategies (select k% of pool -> TabPFN on selected -> fixed val AUC):
  random | tabpfn-entropy (OOF) | raw-kcenter | jepa-kcenter (guided latents)
  | guided-mix (top-2k entropy, k-center to k on guided latents)
Saves figs/budget.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/dead/pfn-jepa/src")
sys.path.insert(0, "/home/dead/playground-series-s6e9")
from pfn_jepa.crossfit import oof_uncertainty
from pfn_jepa.estimator import PFNJEPAClassifier
from pfn_jepa.jepa import embed, prep, train_plug
from src.ev import load, stratified_subsample, tabpfn_predict_proba

sys.path.insert(0, "/home/dead/pfn-jepa/experiments")
from run_matrix import openml_binary

SEED = 0
BUDGETS = [0.05, 0.1, 0.2, 0.4, 1.0]
POOL = 8000


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


def sensitivity_mask_probs(X, y, est) -> np.ndarray:
    return est._sensitivity(X, y)


def run_dataset(name, X, y):
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    # cap pool
    idx = np.random.RandomState(SEED).choice(len(Xpool), min(POOL, len(Xpool)),
                                             replace=False)
    Xpool, ypool = Xpool.iloc[idx], ypool[idx]
    print(f"== {name}: pool={len(ypool)} val={len(yval)} ==", flush=True)

    u = oof_uncertainty(Xpool, ypool, seed=SEED)
    ent = u["entropy"].values
    # guided plug: curriculum + sensitivity masks via a fitted estimator shell
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

    out = []
    for b in BUDGETS:
        k = max(50, int(len(ypool) * b))
        strats = {
            "random": np.random.RandomState(1).choice(len(ypool), k, replace=False),
            "entropy": np.argsort(-ent)[:k],
            "raw-kcenter": kcenter(Zr, k),
            "jepa-kcenter": kcenter(Z, k),
            "guided-mix": None,  # top-2k entropy then kcenter on Z
        }
        top = np.argsort(-ent)[:max(k, min(2000, len(ypool)))]
        lut = {i: r for r, i in enumerate(top)}
        sub = kcenter(Z[top], k)
        strats["guided-mix"] = top[sub]
        for sname, sel in strats.items():
            p = tabpfn_predict_proba(Xpool.iloc[sel], ypool[sel], Xval, seed=SEED)
            a = roc_auc_score(yval, p)
            out.append((name, b, sname, k, a))
            print(f"  b={b} {sname}: {a:.4f}", flush=True)
    return out


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    datasets = {}
    X, y, _, _, _ = load("/home/dead/playground-series-s6e9/data")
    datasets["s6e9"] = (X, y)
    for nm in ["NATICUSdroid", "phoneme"]:
        try:
            datasets[nm] = openml_binary(nm)
        except Exception as e:
            print(f"skip {nm}: {type(e).__name__}", flush=True)
    rows = []
    for name, (X, y) in datasets.items():
        rows += run_dataset(name, X, y)
    pd.DataFrame(rows, columns=["data", "budget", "strategy", "k", "auc"]).to_csv(
        "figs/budget.csv", index=False)
    print(f"saved figs/budget.csv ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
