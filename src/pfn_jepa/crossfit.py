"""Stage 1: cross-fit TabPFN uncertainty. Never in-sample.

5-fold (configurable) TabPFN on a capped pool; per-row OOF proba ->
entropy, margin (1-max), ensemble spread approximated by fold disagreement
(std of the fold-models' proba on that row... each row is OOF for exactly
one fold, so spread comes from a second seed on the same fold-train).
Keeps it to 2 fits per fold: seed A (proba) + seed B (disagreement).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pfn_jepa.data import tabpfn_predict_proba


def oof_uncertainty(X: pd.DataFrame, y: np.ndarray, n_splits=5, seed=0,
                    ctx_cap=12_000):
    """Returns DataFrame indexed like X: p_oof, entropy, margin, disagree."""
    oof = np.zeros(len(X))
    dis = np.zeros(len(X))
    for tr_i, va_i in StratifiedKFold(n_splits, shuffle=True,
                                      random_state=seed).split(X, y):
        Xtr, ytr = X.iloc[tr_i], y[tr_i]
        if len(Xtr) > ctx_cap:  # stratified cap for GPU sanity
            keep = np.concatenate([
                np.random.RandomState(seed).choice(np.where(ytr == k)[0],
                    int(ctx_cap * (ytr == k).mean()), replace=False)
                for k in (0, 1)])
            Xtr, ytr = Xtr.iloc[keep], ytr[keep]
        pa = tabpfn_predict_proba(Xtr, ytr, X.iloc[va_i], seed=seed)
        pb = tabpfn_predict_proba(Xtr, ytr, X.iloc[va_i], seed=seed + 999)
        oof[va_i] = pa
        dis[va_i] = np.abs(pa - pb)
    p = np.clip(oof, 1e-6, 1 - 1e-6)
    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return pd.DataFrame({"p_oof": oof, "entropy": ent,
                         "margin": 1 - np.maximum(p, 1 - p),
                         "disagree": dis}, index=X.index)
