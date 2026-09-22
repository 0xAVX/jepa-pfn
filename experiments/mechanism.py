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

from pfn_jepa.data import kcenter, openml_binary, tabpfn_predict_proba
from pfn_jepa.estimator import PFNJEPAClassifier
from pfn_jepa.jepa import embed, prep, train_plug

SEED = 0


def prep_pool(X, y, pool_n):
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    idx = np.random.RandomState(SEED).choice(len(Xpool), min(pool_n, len(Xpool)),
                                             replace=False)
    return (Xpool.iloc[idx].reset_index(drop=True), ypool[idx],
            Xval, yval)


def seed_idx_of(ypool, frac=0.05, seed=SEED):
    idx = []
    for k in (0, 1):
        kk = np.where(ypool == k)[0]
        idx += np.random.RandomState(seed).choice(
            kk, max(5, int(frac * len(ypool) * (ypool == k).mean())),
            replace=False).tolist()
    return np.array(idx)


def seed_entropy(Xpool, ypool, sidx):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(random_state=SEED)
    clf.fit(Xpool.iloc[sidx], ypool[sidx])
    p = np.clip(clf.predict_proba(Xpool)[:, 1], 1e-6, 1 - 1e-6)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def guided_latents(Xpool, ypool):
    sidx = seed_idx_of(ypool)
    ent = seed_entropy(Xpool, ypool, sidx)
    est = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED)
    est.feats_ = Xpool.columns.tolist()
    sens = est._sensitivity(Xpool.iloc[sidx], ypool[sidx],
                            n=min(500, len(sidx)))
    fp = 0.1 + 0.8 * sens / (sens.max() + 1e-12)
    w = np.ones(len(Xpool))
    w = 1 + 2.0 * (ent / ent.max())
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, row_weights=w, feat_probs=fp,
                          seed=SEED)
    net_u, dev_u = train_plug(Xp, epochs=5, d_lat=32, seed=SEED)
    return embed(net, dev, Xp), embed(net_u, dev_u, Xp), ent, sidx


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
    zg, zu, _ = guided_latents(Xpool, ypool)[:3]
    reps["jepa-uniform"], reps["jepa-guided"] = zu, zg
    sidx = seed_idx_of(ypool)
    unl = np.array([i for i in range(len(ypool)) if i not in set(sidx)])
    n_seed = len(sidx)
    rows = []
    for b in [0.2, 0.4]:
        k = max(n_seed + 25, int(len(ypool) * b))
        need = k - n_seed
        for rname, Z in reps.items():
            sel = np.concatenate([sidx, unl[kcenter(Z[unl], need)]])
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
    sidx = seed_idx_of(y_dirty)
    unl = np.array([i for i in range(len(ypool)) if i not in set(sidx)])
    n_seed, k = len(sidx), 340
    need = k - n_seed

    ent = seed_entropy(Xpool, y_dirty, sidx)
    est = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED)
    est.feats_ = Xpool.columns.tolist()
    sens = est._sensitivity(Xpool.iloc[sidx], y_dirty[sidx],
                            n=min(500, len(sidx)))
    fp = 0.1 + 0.8 * sens / (sens.max() + 1e-12)
    w = 1 + 2.0 * (ent / ent.max())
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, row_weights=w, feat_probs=fp,
                          seed=SEED)
    Z = embed(net, dev, Xp)[unl]
    ent_u = ent[unl]
    pos = np.argsort(-ent_u)
    top = pos[:max(need, min(2000, len(unl)))]
    take = lambda s: np.concatenate([sidx, unl[s]])
    sels = {"random": take(np.random.RandomState(1).choice(len(unl), need,
                                                           replace=False)),
            "entropy": take(pos[:need]),
            "jepa-kcenter": take(kcenter(Z, need)),
            "guided-mix": take(top[kcenter(Z[top], need)])}
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
