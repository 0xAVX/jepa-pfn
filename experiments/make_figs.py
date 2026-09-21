"""README figures: budget curves, ablation ladder, poison bars. Saves figs/*.png."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

b = pd.read_csv("figs/budget.csv")
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, ds in zip(axes, ["s6e9", "NATICUSdroid", "phoneme"]):
    d = b[b.data == ds]
    for s, m in [("random", "o"), ("entropy", "x"), ("jepa-kcenter", "s"),
                 ("guided-mix", "D")]:
        dd = d[d.strategy == s].sort_values("budget")
        ax.plot(dd.budget, dd.auc, marker=m, label=s)
    ax.set_title(ds)
    ax.set_xlabel("label budget")
axes[0].set_ylabel("val AUC")
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig("figs/budget_curves.png", dpi=110)

a = pd.read_csv("figs/ablation.csv")
order = ["tabpfn-raw", "jepa", "+curriculum", "+masks", "+mc4"]
fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True)
for ax, ds in zip(axes.flat, a.data.unique()):
    d = a[a.data == ds].set_index("config").reindex(order)
    ax.bar(range(len(order)), d.auc.values, color=["#2ecc71" if i == 0 else "#3498db"
                                                  for i in range(len(order))])
    ax.set_xticks(range(len(order)), [o.replace("+", "+") for o in order],
                  rotation=20, fontsize=8)
    ax.set_title(ds, fontsize=10)
axes[0, 0].set_ylabel("val AUC")
fig.suptitle("Uniform JEPA hurts; guidance repairs; MC removed")
fig.tight_layout()
fig.savefig("figs/ablation.png", dpi=110)

m = pd.read_csv("figs/mechanism.csv")
p = m[m.exp == "poison"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.5))
ax1.bar(p.strategy, p.auc.values, color="#e74c3c")
ax1.set_title("Poison downstream AUC (same ~8% corruption)")
ax1.tick_params(axis="x", rotation=20, labelsize=8)
ax2.bar(p.strategy, p.corr_frac.values, color="#f39c12")
ax2.set_title("Fraction corrupted selected")
ax2.tick_params(axis="x", rotation=20, labelsize=8)
fig.tight_layout()
fig.savefig("figs/poison.png", dpi=110)
print("saved budget_curves/ablation/poison pngs")
