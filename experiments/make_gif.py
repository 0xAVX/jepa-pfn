"""Demo GIF: phoneme pool, strategies cycle (random/entropy/guided-mix @40%),
selected rings + score. Saves figs/demo.gif."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

D = np.load("figs/demo.npz")
xy, y = D["phoneme/xy"], D["phoneme/y"]
STRATS = ["random", "entropy", "guided-mix"]
B = 0.4
take = np.random.RandomState(0).choice(len(xy), 2500, replace=False)
xs = xy[take]
x0, x1 = xs[:, 0].min(), xs[:, 0].max()
y0, y1 = xs[:, 1].min(), xs[:, 1].max()

fig, ax = plt.subplots(figsize=(5, 5.6))
scat = ax.scatter(xs[:, 0], xs[:, 1], s=4,
                  c=["#e74c3c" if v else "#3498db" for v in y[take]], alpha=0.5)
ring, = ax.plot([], [], "o", ms=9, mfc="none", mec="black", mew=1.5)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_xticks([])
ax.set_yticks([])
title = ax.set_title("")

FRAMES = []
for s in STRATS:
    sel = set(D[f"phoneme/sel_{s}_{B}"].tolist())
    auc = float(D[f"phoneme/auc_{s}_{B}"])
    shown = [g for g in take if g in sel]
    for n in (len(shown) // 3, 2 * len(shown) // 3, len(shown)):
        FRAMES.append((s, auc, shown[:n]))
FRAMES.append(("guided-mix", float(D[f"phoneme/auc_guided-mix_{B}"]),
               [g for g in take if g in set(D[f"phoneme/sel_guided-mix_{B}"].tolist())]))


def draw(f):
    s, auc, shown = FRAMES[f]
    pts = xy[shown] if shown else np.zeros((0, 2))
    ring.set_data(pts[:, 0] if len(pts) else [], pts[:, 1] if len(pts) else [])
    title.set_text(f"{s} @40%: AUC {auc:.4f}")
    return scat, ring, title


FuncAnimation(fig, draw, frames=len(FRAMES), interval=700).save(
    "figs/demo.gif", writer="pillow", dpi=90)
print("saved figs/demo.gif")
