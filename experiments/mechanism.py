"""Mechanism experiments. Saves figs/mechanism.csv.
A. Representation swap (phoneme+NATICUS, budgets 20/40%): same k-center rule on
   raw | PCA | TabPFN-embed (5% seed fit) | JEPA-uniform | JEPA-guided.
B. Poison pool (phoneme, 10% flipped labels): % corrupted selected +
   downstream AUC + clean coverage for random/entropy/jepa-kc/guided-mix.
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
from tabpfn import TabPFNClassifier

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


def prep_pool(X, y, pool_n):
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), min(pool_n, len(Xpool)),
                                             replace=False)
    return (Xpool.iloc[idx].reset_index(drop=True), ypool[idx],
            Xval, yval)


def guided_latents(Xpool, ypool):
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
    net_u, dev_u = train_plug(Xp, epochs=5, d_lat=32, seed=SEED)
    return embed(net, dev, Xp), embed(net_u, dev_u, Xp), ent


def tabpfn_latents(Xpool, ypool, frac=0.05):
    n_seed = max(50, int(len(ypool) * frac))
    sidx = np.random.RandomState(2).choice(len(ypool), n_seed, replace=False)
    clf = TabPFNClassifier(random_state=SEED)
    clf.fit(Xpool.iloc[sidx], ypool[sidx])
    return clf.get_embeddings(Xpool).mean(axis=0)


def exp_swap(name, X, y):
    Xpool, ypool, Xval, yval = prep_pool(X, y, 3404 if name == "phoneme" else 5491)
    Xp, _ = prep(Xpool)
    std = (Xp - Xp.mean(0)) / (Xp.std(0) + 1e-9)
    reps = {"raw": std,
            "pca": PCA(n_components=min(32, Xp.shape[1]),
                       random_state=0).fit_transform(std)}
    reps["tabpfn-emb"] = tabpfn_latents(Xpool, ypool)
    zg, zu, _ = guided_latents(Xpool, ypool)
    reps["jepa-uniform"], reps["jepa-guided"] = zu, zg
    rows = []
    for b in [0.2, 0.4]:
        k = max(50, int(len(ypool) * b))
        for rname, Z in reps.items():
            sel = kcenter(Z, k)
            p = tabpfn_predict_proba(Xpool.iloc[sel], ypool[sel], Xval, seed=SEED)
            a = roc_auc_score(yval, p)
            rows.append(("swap", name, b, rname, a, float("nan")))
            print(f"  swap {name} b={b} {rname}: {a:.4f}", flush=True)
    return rows


def exp_poison(X, y, rate=0.10, budget=0.10):
    Xpool, ypool, Xval, yval = prep_pool(X, y, 3404)
    rng = np.random.RandomState(7)
    corr = np.zeros(len(ypool), bool)
    corr[rng.choice(len(ypool), int(rate * len(ypool)), replace=False)] = True
    y_dirty = ypool.copy()
    y_dirty[corr] = 1 - y_dirty[corr]
    k = max(50, int(len(ypool) * budget))

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
    Z = embed(net, dev, Xp)
    top = np.argsort(-ent)[:max(k, min(2000, len(ypool)))]
    sels = {"random": np.random.RandomState(1).choice(len(ypool), k, replace=False),
            "entropy": np.argsort(-ent)[:k],
            "jepa-kcenter": kcenter(Z, k),
            "guided-mix": top[kcenter(Z[top], k)]}
    rows = []
    for sname, sel in sels.items():
        frac_corr = float(corr[sel].mean())
        clean_cov = float((~corr[sel]).sum() / (~corr).sum())
        p = tabpfn_predict_proba(Xpool.iloc[sel], y_dirty[sel], Xval, seed=SEED)
        a = roc_auc_score(yval, p)
        rows.append(("poison", "phoneme", budget, sname, a, frac_corr))
        print(f"  poison {sname}: corr_sel={frac_corr:.3f} clean_cov={clean_cov:.3f} "
              f"AUC={a:.4f}", flush=True)
    return rows


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    rows = []
    datasets = {"phoneme": openml_binary("phoneme"),
                "NATICUSdroid": openml_binary("NATICUSdroid")}
    for name, (X, y) in datasets.items():
        print(f"== swap {name} ==", flush=True)
        rows += exp_swap(name, X, y)
    print("== poison phoneme ==", flush=True)
    rows += exp_poison(*datasets["phoneme"])
    pd.DataFrame(rows, columns=["exp", "data", "budget", "strategy", "auc",
                                "corr_frac"]).to_csv("figs/mechanism.csv", index=False)
    print(f"saved figs/mechanism.csv ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
