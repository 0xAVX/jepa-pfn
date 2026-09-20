"""JEPA core with PFN-guided hooks (row curriculum + feature mask priors).

Superset of the tabpfn-jepa-plug encoder: same context/target EMA + VICReg
anti-collapse, plus:
  row_weights: sampling probs over rows (PFN uncertainty curriculum).
  feat_probs:  P(feature j is masked from context) (PFN sensitivity masks).
diagnostics(): per-row recon error, completion variance, group errors —
  the "how surprising is this row" features fed back to TabPFN.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


def prep(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = df.columns.tolist()
    out = np.zeros((len(df), len(cols)), dtype=np.float32)
    for j, c in enumerate(cols):
        v = df[c]
        if str(v.dtype) in ("category", "object"):
            out[:, j] = pd.factorize(v.astype(str))[0].astype(np.float32)
        else:
            v = pd.to_numeric(v, errors="coerce").fillna(0).astype(np.float32)
            out[:, j] = (v - float(v.mean())) / (float(v.std()) + 1e-6)
    return out, cols


class Plug(nn.Module):
    def __init__(self, d_in: int, d_lat: int = 128, depth: int = 2, width: int = 256):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.GELU()]
            d = width
        layers.append(nn.Linear(d, d_lat))
        self.ctx = nn.Sequential(*layers)
        self.pred = nn.Sequential(nn.Linear(d_lat, width), nn.GELU(),
                                  nn.Linear(width, d_lat))

    def forward(self, x_ctx, x_tgt, mask_ctx, mask_tgt):
        z_ctx = self.ctx(x_ctx * mask_ctx)
        with torch.no_grad():
            z_tgt = self.tgt(x_tgt * mask_tgt)
        return self.pred(z_ctx), z_tgt

    def attach_target(self):
        self.tgt = copy.deepcopy(self.ctx)
        for p in self.tgt.parameters():
            p.requires_grad = False


def vicreg(z: torch.Tensor, var_w=1.0, cov_w=0.04):
    std = z.std(dim=0) + 1e-4
    var_loss = torch.relu(1 - std).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (len(z) - 1)
    off = cov - torch.diag(torch.diag(cov))
    return var_w * var_loss + cov_w * (off ** 2).mean()


def _draw_masks(shape, feat_probs, g, dev):
    """feat_probs[j] = P(feature j masked from context)."""
    u = torch.rand(shape, generator=g)
    return (u < torch.as_tensor(feat_probs, dtype=torch.float32)).float().to(dev)


def train_plug(X: np.ndarray, epochs=15, batch=2048, d_lat=128, depth=2, width=256,
               mask_ratio=0.4, row_weights=None, feat_probs=None, seed=0,
               device="auto", verbose=False):
    g = torch.Generator().manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if device == "auto" else torch.device(device)
    d = X.shape[1]
    if feat_probs is None:
        feat_probs = np.full(d, mask_ratio)
    feat_probs = np.asarray(feat_probs, dtype=np.float64)
    if row_weights is None:
        row_weights = np.ones(len(X)) / len(X)
    row_weights = np.asarray(row_weights, dtype=np.float64)
    row_weights /= row_weights.sum()

    net = Plug(d, d_lat, depth, width).to(dev)
    net.attach_target()
    opt = torch.optim.AdamW(list(net.ctx.parameters()) + list(net.pred.parameters()),
                            lr=3e-3, weight_decay=1e-4)
    Xt = torch.from_numpy(X)
    n = len(X)
    for ep in range(epochs):
        perm = np.random.RandomState(seed + ep).choice(n, n, p=row_weights)
        perm = torch.from_numpy(perm)
        tot, nb = 0.0, 0
        for i in range(0, n, batch):
            xb = Xt[perm[i:i + batch]].to(dev)
            m_tgt = _draw_masks(xb.shape, feat_probs, g, dev)
            m_ctx = 1 - m_tgt
            pred, tgt = net(xb, xb, m_ctx, m_tgt)
            loss = float(nn.functional.mse_loss(pred, tgt)) + vicreg(pred) \
                + vicreg(net.ctx(xb * m_ctx))
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for pc, pt in zip(net.ctx.parameters(), net.tgt.parameters()):
                    pt.mul_(0.99).add_(pc, alpha=0.01)
            tot += float(loss.detach()); nb += 1
        if verbose:
            print(f"ep {ep}: {tot / nb:.4f}", flush=True)
    net.eval()
    return net, dev


@torch.no_grad()
def relevance(net: Plug, dev, X: np.ndarray, batch=4096, seed=0, mask_ratio=0.4,
              ) -> np.ndarray:
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X).to(dev)
    d = X.shape[1]
    allm = (torch.rand(min(len(X), batch * 4), d, generator=g) > mask_ratio).float()
    xb0 = Xt[:len(allm)]
    p0, t0 = net(xb0, xb0, allm.to(dev), (1 - allm).to(dev))
    base = float(nn.functional.mse_loss(p0, t0))
    rel = np.zeros(d)
    for j in range(d):
        m = allm.clone()
        m[:, j] = 0
        p, t = net(xb0, xb0, m.to(dev), (1 - m).to(dev))
        rel[j] = float(nn.functional.mse_loss(p, t)) - base
    return np.maximum(rel, 0)


@torch.no_grad()
def embed(net: Plug, dev, X: np.ndarray, batch=8192) -> np.ndarray:
    Xt = torch.from_numpy(X).to(dev)
    ones = torch.ones(1, X.shape[1]).to(dev)
    return torch.cat([net.ctx(Xt[i:i + batch] * ones) for i in range(0, len(Xt), batch)]
                     ).cpu().numpy()


@torch.no_grad()
def diagnostics(net: Plug, dev, X: np.ndarray, n_groups=4, n_draws=8, seed=0,
                ) -> pd.DataFrame:
    """Per-row JEPA surprise features: recon error, completion variance,
    per-group errors. Everything TabPFN gets told about manifold fit."""
    g = torch.Generator().manual_seed(seed)
    Xt = torch.from_numpy(X).to(dev)
    n, d = X.shape
    groups = np.array_split(np.arange(d), n_groups)
    rec = np.zeros(n)
    draws = np.zeros((n, n_draws))
    grp = np.zeros((n, n_groups))
    BS = 4096
    for s in range(0, n, BS):
        xb = Xt[s:s + BS]
        nb = len(xb)
        m = (torch.rand(nb, d, generator=g) > 0.4).float().to(dev)
        p, t = net(xb, xb, m, 1 - m)
        rec[s:s + nb] = ((p - t) ** 2).mean(dim=1).cpu().numpy()
        for k in range(n_draws):
            mk = (torch.rand(nb, d, generator=g) > 0.4).float().to(dev)
            pk, _ = net(xb, xb, mk, 1 - mk)
            draws[s:s + nb, k] = pk.norm(dim=1).cpu().numpy()
        for gi, idx in enumerate(groups):
            mg = torch.ones(nb, d).to(dev)
            mg[:, idx] = 0
            pg, tg = net(xb, xb, mg, 1 - mg)
            grp[s:s + nb, gi] = ((pg - tg) ** 2).mean(dim=1).cpu().numpy()
    out = {"jepa_recon": rec, "jepa_completion_var": draws.var(axis=1)}
    for gi in range(n_groups):
        out[f"jepa_group_err_{gi}"] = grp[:, gi]
    return pd.DataFrame(out)
