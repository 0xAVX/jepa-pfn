"""Ablation ladder x datasets -> figs/ablation.csv.
Configs: raw | jepa | +curriculum | +masks | +diag(full loop) | +mc4
Datasets: S6E9-sub, NATICUSdroid, phoneme, churn (skipped gracefully if download fails).
Caps keep the 8GB card happy. Usage: <venv-python> experiments/run_matrix.py
"""
from __future__ import annotations

import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from pfn_jepa.data import load_s6e9, openml_binary, stratified_subsample, tabpfn_predict_proba
from pfn_jepa.estimator import PFNJEPAClassifier

SEED = 0

CONFIGS = [
    ("tabpfn-raw", {}),
    ("jepa", {"curriculum": False, "guided_masks": False}),
    ("+curriculum", {"guided_masks": False}),
    ("+masks", {}),
    ("+mc4", {"mc_samples": 4}),
]


def get_datasets():
    out = {}
    try:
        X, y, _, _, _ = load_s6e9("data")
        Xs, ys = stratified_subsample(X, y, 12_000, seed=SEED)
        out["s6e9-12k"] = (Xs, ys)
    except Exception as e:
        print("skip s6e9:", e, flush=True)
    for name in ["NATICUSdroid", "phoneme", "churn"]:
        try:
            out[name] = openml_binary(name)
        except Exception as e:
            print(f"skip {name}: {type(e).__name__}", flush=True)
    return out


def run_one(name, X, y):
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=min(8_000, len(y) // 3),
                                          stratify=y, random_state=1)
    print(f"== {name}: train={len(ytr)} val={len(yva)} ==", flush=True)
    res = {}
    p = tabpfn_predict_proba(Xtr, ytr, Xva, seed=SEED)
    res["tabpfn-raw"] = roc_auc_score(yva, p)
    print(f"  tabpfn-raw: {res['tabpfn-raw']:.4f}", flush=True)
    for cfg, kw in CONFIGS[1:]:
        try:
            m = PFNJEPAClassifier(d_lat=32, epochs=5, seed=SEED, **kw)
            m.fit(Xtr, ytr)
            p = m.predict_proba(Xva)[:, 1]
            res[cfg] = roc_auc_score(yva, p)
        except Exception as e:
            res[cfg] = float("nan")
            print(f"  {cfg}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
            traceback.print_exc(limit=3)
        else:
            print(f"  {cfg}: {res[cfg]:.4f}", flush=True)
    return res


def main():
    t0 = time.time()
    Path("figs").mkdir(exist_ok=True)
    rows = []
    for name, (X, y) in get_datasets().items():
        res = run_one(name, X, y)
        for cfg, auc in res.items():
            rows.append((name, cfg, auc))
    df = pd.DataFrame(rows, columns=["data", "config", "auc"])
    df.to_csv("figs/ablation.csv", index=False)
    print(df.pivot(index="data", columns="config", values="auc").round(4).to_string(),
          flush=True)
    print(f"saved figs/ablation.csv ({(time.time()-t0)/60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
