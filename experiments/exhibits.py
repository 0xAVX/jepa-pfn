"""Exhibits: MADELON feature retention + APS rare-event acquisition.
Saves figs/exhibits.csv. GPU for TabPFN/JEPA, CPU for the rest.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/home/dead/pfn-jepa/src")
sys.path.insert(0, "/home/dead/pfn-jepa/experiments")
sys.path.insert(0, "/home/dead/playground-series-s6e9")
from pfn_jepa.crossfit import oof_uncertainty
from pfn_jepa.estimator import PFNJEPAClassifier
from pfn_jepa.jepa import embed, prep, relevance, train_plug
from selection import kcenter
from src.ev import tabpfn_predict_proba

SEED = 0
rng = np.random.RandomState(0)


def load_madelon():
    d = "data/madelon/MADELON"
    Xtr = pd.read_csv(f"{d}/madelon_train.data", sep=r"\s+", header=None)
    ytr = pd.read_csv(f"{d}/madelon_train.labels", header=None)[0].values
    Xva = pd.read_csv(f"{d}/madelon_valid.data", sep=r"\s+", header=None)
    yva = pd.read_csv("data/madelon/madelon_valid.labels", header=None)[0].values
    ytr = (ytr == 1).astype(int)
    yva = (yva == 1).astype(int)
    Xtr.columns = [f"f{i}" for i in range(Xtr.shape[1])]
    Xva.columns = Xtr.columns
    return Xtr, ytr, Xva, yva


def madelon_curve():
    print("== MADELON feature retention ==", flush=True)
    Xtr, ytr, Xva, yva = load_madelon()
    Xp, cols = prep(Xtr)
    net, dev = train_plug(Xp, epochs=10, seed=SEED, verbose=True)
    order = np.argsort(-relevance(net, dev, Xp))
    rows = []
    p = tabpfn_predict_proba(Xtr, ytr, Xva, seed=SEED)
    rows.append(("madelon", 500, "all", roc_auc_score(yva, p)))
    print(f"  all500: {rows[-1][3]:.4f}", flush=True)
    for k in [10, 20, 50, 100, 250]:
        pj = tabpfn_predict_proba(Xtr.iloc[:, order[:k]], ytr,
                                  Xva.iloc[:, order[:k]], seed=SEED)
        rk = rng.choice(Xtr.shape[1], k, replace=False)
        pr = tabpfn_predict_proba(Xtr.iloc[:, rk], ytr, Xva.iloc[:, rk], seed=SEED)
        pb = tabpfn_predict_proba(Xtr.iloc[:, order[-k:]], ytr,
                                  Xva.iloc[:, order[-k:]], seed=SEED)
        rows.append(("madelon", k, "jepa-top", roc_auc_score(yva, pj)))
        rows.append(("madelon", k, "random", roc_auc_score(yva, pr)))
        rows.append(("madelon", k, "jepa-bottom", roc_auc_score(yva, pb)))
        print(f"  k={k}: top {rows[-3][3]:.4f} vs random {rows[-2][3]:.4f} "
              f"vs bottom {rows[-1][3]:.4f}", flush=True)
    return rows


def load_aps():
    import zipfile
    zf = zipfile.ZipFile("data/aps.zip")
    names = zf.namelist()
    tr = [n for n in names if "train" in n.lower()][0]
    te = [n for n in names if "test" in n.lower() and n.endswith(".csv")]
    dtr = pd.read_csv(zf.open(tr), na_values=["na"], skiprows=20)
    dte = pd.read_csv(zf.open(te[0]), na_values=["na"], skiprows=20) if te else None
    ytr = (dtr["class"] == "pos").astype(int).values
    Xtr = dtr.drop(columns=["class"])
    if dte is not None and "class" in dte.columns:
        yte = (dte["class"] == "pos").astype(int).values
        Xte = dte.drop(columns=["class"])
    else:
        Xtr, Xte, ytr, yte = train_test_split(Xtr, ytr, test_size=16000,
                                              stratify=ytr, random_state=1)
    for c in Xtr.columns:
        if Xtr[c].dtype == object:
            Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce")
            Xte[c] = pd.to_numeric(Xte[c], errors="coerce")
    return (Xtr.reset_index(drop=True), ytr,
            Xte.reset_index(drop=True), yte)


def aps_acquire():
    print("== APS rare-event acquisition ==", flush=True)
    Xtr, ytr, Xte, yte = load_aps()
    print(f"train={Xtr.shape} pos_rate={ytr.mean():.4f} test={Xte.shape}",
          flush=True)
    # pool 20k from train, fixed test
    idx = np.random.RandomState(SEED).choice(len(Xtr), 20000, replace=False)
    Xpool, ypool = Xtr.iloc[idx].reset_index(drop=True), ytr[idx]
    u = oof_uncertainty(Xpool, ypool, seed=SEED)
    ent = u["entropy"].values
    est = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED)
    est.feats_ = Xpool.columns.tolist()
    sens = est._sensitivity(Xpool.fillna(0), ypool, n=300)
    fp = 0.1 + 0.8 * sens / (sens.max() + 1e-12)
    w = 1 + 2.0 * (ent / ent.max())
    Xp, _ = prep(Xpool.fillna(0))
    net, dev = train_plug(Xp, epochs=5, d_lat=32, row_weights=w, feat_probs=fp,
                          seed=SEED)
    Z = embed(net, dev, Xp)
    rows = []
    for b in [0.01, 0.02, 0.05, 0.10]:
        k = max(50, int(len(ypool) * b))
        top = np.argsort(-ent)[:max(k, min(2000, len(ypool)))]
        sels = {"random": rng.choice(len(ypool), k, replace=False),
                "entropy": np.argsort(-ent)[:k],
                "jepa-kcenter": kcenter(Z, k),
                "guided-mix": top[kcenter(Z[top], k)]}
        for sname, sel in sels.items():
            p = tabpfn_predict_proba(Xpool.iloc[sel], ypool[sel], Xte, seed=SEED)
            pr = average_precision_score(yte, p)
            rec = float(((p > 0.5).astype(int) & yte).sum() / max(yte.sum(), 1))
            found = int(ypool[sel].sum())
            rows.append(("aps", b, sname, pr, rec, found))
            print(f"  b={b} {sname}: PR-AUC={pr:.4f} recall={rec:.3f} "
                  f"pos_found={found}", flush=True)
    return rows


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    fr = madelon_curve()
    pd.DataFrame(fr, columns=["data", "k", "setup", "auc"]).to_csv(
        "figs/exhibits_feat.csv", index=False)
    ar = aps_acquire()
    pd.DataFrame(ar, columns=["data", "budget", "strategy", "pr_auc", "recall",
                              "pos_found"]).to_csv("figs/exhibits_aps.csv", index=False)
    print(f"saved exhibits ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
