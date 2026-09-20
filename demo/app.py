"""PFN-JEPA demo: spend labels where they matter. Deterministic: loads
figs/demo.npz (+ ablation.csv); nothing retrains. Run: <venv-python> demo/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, request, render_template_string

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/home/dead/pfn-jepa/src")

D = np.load(ROOT / "figs" / "demo.npz")
DATASETS = ["phoneme", "s6e9", "NATICUSdroid"]
STRATS = ["random", "entropy", "raw-kcenter", "jepa-kcenter", "guided-mix"]
BUDGETS = [0.05, 0.1, 0.2, 0.4]

ABL = [
    ("Raw TabPFN", "baseline", ""),
    ("Uniform JEPA augmentation", "✗ hurt on all 4 datasets", "bad"),
    ("Guided curriculum", "✓ repaired 3/4", "good"),
    ("Guided feature masks", "✓ repaired 3/4", "good"),
    ("MC completion", "✗ removed after −0.015 on phoneme", "bad"),
    ("Guided-mix selection", "✓ only method beating random on hard data", "good"),
    ("Feature selection", "✓ +0.086 over random on NATICUS", "good"),
]

app = Flask(__name__)

PAGE = """
<h1>PFN-JEPA — spend labels where they matter</h1>
<form method=get>
Dataset: <select name=d>{% for x in ds %}<option {{'selected' if x==d}}>{{x}}</option>{% endfor %}</select>
Budget: <select name=b>{% for x in bs %}<option {{'selected' if x==b}}>{{x}}</option>{% endfor %}</select><br>
{% for s in ss %}<input type=radio name=s value="{{s}}" {{'checked' if s==sel}}> {{s }} {% endfor %}<br>
<input type=submit value="Show">
</form>
<h2>{{sel}} @ {{b}} → {{'%.4f' % score}} &nbsp; (random {{'%.4f' % base}}, Δ {{'%+.4f' % (score-base)}})</h2>
{{svg|safe}}
<h3>Why did this work?</h3>
<p>TabPFN identifies difficult regions. JEPA learns their structure. Diversity prevents
repeatedly choosing the same kind of difficult example. Entropy alone over-concentrates
on unlearnable rows — toggle to entropy to see it.</p>
<h3>What we found</h3>
<ul>{% for n, v, c in abl %}<li class="{{c}}"><b>{{n}}</b> — {{v}}</li>{% endfor %}</ul>
<style>.bad{color:#a00}.good{color:#0a0}</style>
"""


def scatter(ds, b, sel):
    xy, y = D[f"{ds}/xy"], D[f"{ds}/y"]
    n = len(xy)
    take = np.random.RandomState(0).choice(n, min(n, 2500), replace=False)
    xs = xy[take]
    x0, x1 = xs[:, 0].min(), xs[:, 0].max()
    y0, y1 = xs[:, 1].min(), xs[:, 1].max()
    W = 520
    sc = lambda a, lo, hi: 10 + (W - 20) * (a - lo) / max(hi - lo, 1e-9)
    picked = set(D[f"{ds}/sel_{sel}_{b}"].tolist())
    out = [f'<svg width="{W}" height="{W}" style="border:1px solid #ccc">']
    for i, g in enumerate(take):
        col = "#e74c3c" if y[g] else "#3498db"
        r, extra = (2.0, "") if g not in picked else (
            4.5, ' stroke="black" stroke-width="1.5"')
        out.append(f'<circle cx="{sc(xs[i,0],x0,x1):.1f}" cy="{sc(xs[i,1],y0,y1):.1f}"'
                   f' r="{r}" fill="{col}"{extra}/>')
    out.append("</svg><p>● class 1 ● class 0 (ringed = selected for labeling)</p>")
    return "".join(out)


@app.get("/")
def index():
    ds = request.args.get("d", "phoneme")
    b = float(request.args.get("b", 0.4))
    sel = request.args.get("s", "guided-mix")
    if ds not in DATASETS:
        ds = "phoneme"
    score = float(D[f"{ds}/auc_{sel}_{b}"])
    base = float(D[f"{ds}/auc_random_{b}"])
    return render_template_string(PAGE, ds=DATASETS, bs=BUDGETS, ss=STRATS, d=ds, b=b,
                                  sel=sel, score=score, base=base,
                                  svg=scatter(ds, b, sel), abl=ABL)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
