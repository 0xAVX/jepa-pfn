"""PFNJEPAClassifier: the closed loop in one sklearn-style API.

Stages (all on by default; each ablatable by flag):
  plain loop:      JEPA trained uniform -> raw + latents + diagnostics -> TabPFN
  curriculum:      JEPA row-sampling ∝ 1 + lam * OOF entropy
  guided masks:    JEPA feature-mask probs ∝ TabPFN shuffle-sensitivity
  mc completions:  M stochastic JEPA completions -> TabPFN proba average
explain_uncertainty(): per-row TabPFN entropy, JEPA surprise, fragile feats,
  completion disagreement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .crossfit import oof_uncertainty
from .data import tabpfn_predict_proba
from .jepa import diagnostics, embed, prep, prep_apply, prep_fit, relevance, train_plug


class PFNJEPAClassifier:
    def __init__(self, d_lat=64, epochs=10, lam=2.0, mask_ratio=0.4,
                 curriculum=True, guided_masks=True, use_diagnostics=True,
                 use_latents=True, mc_samples=0, seed=0):
        self.d_lat, self.epochs, self.lam = d_lat, epochs, lam
        self.mask_ratio = mask_ratio
        self.curriculum, self.guided_masks = curriculum, guided_masks
        self.use_diagnostics, self.use_latents = use_diagnostics, use_latents
        self.mc_samples, self.seed = mc_samples, seed

    def _sensitivity(self, X: pd.DataFrame, y: np.ndarray, n=1000) -> np.ndarray:
        rng = np.random.RandomState(self.seed)
        idx = rng.choice(len(X), min(n, len(X)), replace=False)
        Xs, ys = X.iloc[idx], y[idx]
        p0 = tabpfn_predict_proba(Xs, ys, Xs, seed=self.seed)
        s = np.zeros(X.shape[1])
        for j, c in enumerate(X.columns):
            Xp = Xs.copy()
            Xp[c] = Xp[c].sample(frac=1, random_state=j).values
            s[j] = np.abs(tabpfn_predict_proba(Xs, ys, Xp, seed=self.seed) - p0).mean()
        return s / (s.sum() + 1e-12)

    def fit(self, X: pd.DataFrame, y):
        y = np.asarray(y)
        X = X.reset_index(drop=True)
        self.feats_ = X.columns.tolist()
        # Stage 1: OOF uncertainty (skip if plain loop requested)
        if self.curriculum or self.guided_masks:
            u = oof_uncertainty(X, y, seed=self.seed)
            self.oof_ = u
            w = 1 + self.lam * (u["entropy"] / u["entropy"].max()).values
        else:
            w = None
        sens = self._sensitivity(X, y) if self.guided_masks else None
        fp = None if sens is None else (0.1 + 0.8 * sens / (sens.max() + 1e-12))
        # Stage 2: JEPA with PFN-guided curriculum
        Xp, _ = prep(X)
        self.prep_ = prep_fit(X)
        Xp, _ = prep_apply(X, self.prep_)
        self.net_, self.dev_ = train_plug(
            Xp, epochs=self.epochs, d_lat=self.d_lat, mask_ratio=self.mask_ratio,
            row_weights=w, feat_probs=fp, seed=self.seed)
        self.rel_ = relevance(self.net_, self.dev_, Xp, mask_ratio=self.mask_ratio)
        # Stage 3: augmented frame + final TabPFN fit
        self.train_aug_ = self._augment(X, Xp)
        self.y_ = y
        return self

    def _augment(self, X: pd.DataFrame, Xp: np.ndarray) -> pd.DataFrame:
        parts = [X.reset_index(drop=True)]
        if self.use_latents:
            E = embed(self.net_, self.dev_, Xp)
            parts.append(pd.DataFrame(E, columns=[f"jepa{i}" for i in range(E.shape[1])]))
        if self.use_diagnostics:
            parts.append(diagnostics(self.net_, self.dev_, Xp).reset_index(drop=True))
        return pd.concat(parts, axis=1)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reset_index(drop=True)
        Xp, _ = prep_apply(X[self.feats_], self.prep_)
        Xa = self._augment(X[self.feats_], Xp)
        # align columns to train augmentation
        for c in self.train_aug_.columns:
            if c not in Xa.columns:
                Xa[c] = 0
        Xa = Xa[self.train_aug_.columns]
        p = tabpfn_predict_proba(self.train_aug_, self.y_, Xa, seed=self.seed)
        if self.mc_samples > 0:
            import torch
            ps = [p]
            lat_cols = [c for c in Xa.columns if c.startswith("jepa")
                        and not c.startswith("jepa_")]
            Xt = torch.from_numpy(Xp).to(self.dev_)
            g = torch.Generator().manual_seed(self.seed)
            for m in range(self.mc_samples):
                mk = (torch.rand(Xt.shape, generator=g) > 0.2).float().to(self.dev_)
                with torch.no_grad():
                    Em = self.net_.ctx(Xt * mk).cpu().numpy()
                Xm = Xa.copy()
                Xm[lat_cols] = Em[:, :len(lat_cols)]
                ps.append(tabpfn_predict_proba(self.train_aug_, self.y_, Xm,
                                               seed=self.seed))
            self.last_disagreement_ = float(np.std(ps, axis=0).mean())
            p = np.mean(ps, axis=0)
        return np.vstack([1 - p, p]).T

    def explain_uncertainty(self, X: pd.DataFrame) -> dict:
        X = X.iloc[[0]].reset_index(drop=True)
        Xp, _ = prep_apply(X[self.feats_], self.prep_)
        p = self.predict_proba(X)[0, 1]
        ent = float(-(p * np.log(p + 1e-9) + (1 - p) * np.log(1 - p + 1e-9)))
        d = diagnostics(self.net_, self.dev_, Xp).iloc[0].to_dict()
        order = np.argsort(-self.rel_)
        return {"tabpfn_entropy": ent, "jepa_surprise": float(d["jepa_recon"]),
                "fragile_features": [self.feats_[i] for i in order[:3]],
                "completion_disagreement": float(getattr(self, "last_disagreement_", 0.0))}
