"""Selection baselines under a real seed protocol (no pool-label leakage).

5% stratified seed labeled. TabPFN fits on seed only; entropy is predicted on
the unlabeled pool; JEPA trains on X only (label-free); sensitivity uses the
seed. Budgets count total labels (seed included): to reach k, acquire k-n_seed
more. Strategies: random | seed-entropy | raw-kcenter | jepa-kcenter (guided
latents) | guided-mix. Saves figs/budget.csv.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pfn_jepa.data import kcenter, load_s6e9, openml_binary, stratified_subsample, tabpfn_predict_proba
from pfn_jepa.estimator import PFNJEPAClassifier
from pfn_jepa.jepa import embed, prep, train_plug

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


def run_dataset(name, X, y):
    Xpool, Xval, ypool, yval = train_test_split(X, y, test_size=2000,
                                                stratify=y, random_state=1)
    # cap pool
    idx = np.random.RandomState(SEED).choice(len(Xpool), min(POOL, len(Xpool)),
                                             replace=False)
    Xpool, ypool = Xpool.iloc[idx].reset_index(drop=True), ypool[idx]
    print(f"== {name}: pool={len(ypool)} val={len(yval)} ==", flush=True)

    # seed protocol: 5% labeled, everything else ranks the unlabeled pool
    n_seed = max(50, int(0.05 * len(ypool)))
    seed_idx = []
    for k in (0, 1):
        kk = np.where(ypool == k)[0]
        seed_idx += np.random.RandomState(SEED).choice(
            kk, max(5, int(n_seed * (ypool == k).mean())),
            replace=False).tolist()
    seed_idx = np.array(seed_idx)
    unl = np.array([i for i in range(len(ypool)) if i not in set(seed_idx)])
    Xs, ys = Xpool.iloc[seed_idx], ypool[seed_idx]
    from tabpfn import TabPFNClassifier
    seed_clf = TabPFNClassifier(random_state=SEED)
    seed_clf.fit(Xs, ys)
    ent_all = -(lambda p: p * np.log(p) + (1 - p) * np.log(1 - p))(
        np.clip(seed_clf.predict_proba(Xpool)[:, 1], 1e-6, 1 - 1e-6))
    ent = ent_all[unl]
    # guided plug: curriculum + sensitivity masks from seed-derived signals;
    # JEPA itself sees X only (label-free)
    est = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED)
    est.feats_ = Xpool.columns.tolist()
    sens = est._sensitivity(Xs, ys, n=min(500, len(ys)))
    fp = 0.1 + 0.8 * sens / (sens.max() + 1e-12)
    w = np.ones(len(Xpool))
    w[unl] = 1 + 2.0 * (ent / ent.max())
    Xp, _ = prep(Xpool)
    net, dev = train_plug(Xp, epochs=5, d_lat=32, row_weights=w, feat_probs=fp,
                          seed=SEED)
    Z = embed(net, dev, Xp)[unl]
    Zr = (Xp / (np.abs(Xp).max(axis=0) + 1e-9))[unl]

    out = []
    pos_ent = np.argsort(-ent)  # positions into unl-ordered arrays
    for b in BUDGETS:
        k = max(n_seed + 25, int(len(ypool) * b))
        need = k - n_seed
        take = lambda pos: unl[pos]  # back to pool indices
        strats = {
            "random": np.concatenate(
                [seed_idx, take(np.random.RandomState(1).choice(len(unl), need,
                                                                replace=False))]),
            "entropy": np.concatenate([seed_idx, take(pos_ent[:need])]),
            "raw-kcenter": np.concatenate([seed_idx, take(kcenter(Zr, need))]),
            "jepa-kcenter": np.concatenate([seed_idx, take(kcenter(Z, need))]),
        }
        top = pos_ent[:max(need, min(2000, len(unl)))]
        sub = kcenter(Z[top], need)
        strats["guided-mix"] = np.concatenate([seed_idx, take(top[sub])])
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
    try:
        X, y, _, _, _ = load_s6e9("data")
        datasets["s6e9"] = (X, y)
    except Exception as e:
        print("skip s6e9:", e, flush=True)
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
