"""PFN-JEPA demo v2: spend labels / why-not-entropy / map-vs-compass.
Deterministic: loads figs/demo.npz + figs/poison_demo.npz; nothing retrains.
Run: <venv-python> demo/app.py  (port 5001)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from flask import Flask, request, render_template_string

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, "/home/dead/pfn-jepa/src")

D = np.load(ROOT / "figs" / "demo.npz")
P = np.load(ROOT / "figs" / "poison_demo.npz") if (ROOT / "figs" / "poison_demo.npz").exists() else None
DATASETS = ["phoneme", "s6e9", "NATICUSdroid"]
STRATS = ["random", "entropy", "raw-kcenter", "jepa-kcenter", "guided-mix"]
BUDGETS = [0.05, 0.1, 0.2, 0.4]
PSTRATS = ["random", "entropy", "jepa-uniform", "jepa-guided", "guided-mix"]

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
NAV = '<p><a href="/?view=spend">Spend labels</a> | <a href="/?view=poison">Why not entropy?</a> | <a href="/?view=map">Map vs Compass</a></p>'
STYLE = "<style>.bad{color:#a00}.good{color:#0a0}svg{border:1px solid #ccc}</style>"

SPEND = NAV + """
<h1>Spend your labels (phoneme default)</h1>
<form method=get><input type=hidden name=view value=spend>
Dataset: <select name=d>{% for x in ds %}<option {{'selected' if x==d}}>{{x}}</option>{% endfor %}</select>
Budget: <select name=b>{% for x in bs %}<option {{'selected' if x==b}}>{{x}}</option>{% endfor %}</select><br>
{% for s in ss %}<input type=radio name=s value="{{s}}" {{'checked' if s==sel}}> {{s}} {% endfor %}<br>
<input type=submit value="Show"></form>
<h2>{{sel}} @ {{b}} → {{'%.4f' % score}} (random {{'%.4f' % base}}, Δ {{'%+.4f' % (score-base)}})</h2>
{{svg|safe}}
<h3>What we found</h3>
<ul>{% for n, v, c in abl %}<li class="{{c}}"><b>{{n}}</b> — {{v}}</li>{% endfor %}</ul>
""" + STYLE

POISON = NAV + """
<h1>Why not entropy? — poison stress test (phoneme, 10% flipped labels)</h1>
<form method=get><input type=hidden name=view value=poison>
Strategy: <select name=s>{% for x in ss %}<option {{'selected' if x==s}}>{{x}}</option>{% endfor %}</select>
Overlay: <select name=o><option value="sel" {{'selected' if o=='sel'}}>selections</option>
<option value="corr" {{'selected' if o=='corr'}}>corrupted points</option></select>
<input type=submit value="Show"></form>
<h2>Corrupted selected</h2>
<ul>{% for s, f, a in rows %}<li><b>{{s}}</b>: {{'%.1f' % (100*f)}}% corrupted → AUC {{'%.3f' % a}}</li>{% endfor %}</ul>
{{svg|safe}}
<p><b>The reveal:</b> every strategy selects ~8% corruption — yet entropy collapses to
{{'%.3f' % ent_auc}} while diversity holds {{'%.3f' % div_auc}}+.
Failure is <i>where</i> corruption lands (fragile low-margin regions), not <i>how much</i>.</p>
""" + STYLE

MAP = NAV + """
<h1>Map vs Compass</h1>
<form method=get><input type=hidden name=view value=map>
Representation: <select name=r><option {{'selected' if r=='jepa-uniform'}}>jepa-uniform</option>
<option {{'selected' if r=='jepa-guided'}}>jepa-guided</option></select>
<input type=submit value="Show"></form>
{{svg|safe}}
<p><b>Uniform JEPA</b> — best coverage geometry (the MAP: where haven't we looked?).<br>
<b>Guided JEPA</b> — task-aligned ranking (the COMPASS: where might learning matter?).<br>
<b>Guided-mix</b> = compass rank × map diversity → the only strategy beating random on hard data.</p>
""" + STYLE


def scatter(xy, y, picked=None, corr=None, overlay="sel", W=520):
    n = len(xy)
    take = np.random.RandomState(0).choice(n, min(n, 2500), replace=False)
    xs = xy[take]
    x0, x1 = xs[:, 0].min(), xs[:, 0].max()
    y0, y1 = xs[:, 1].min(), xs[:, 1].max()
    sc = lambda a, lo, hi: 10 + (W - 20) * (a - lo) / max(hi - lo, 1e-9)
    picked = set() if picked is None else set(np.asarray(picked).tolist())
    out = [f'<svg width="{W}" height="{W}">']
    for i, g in enumerate(take):
        if overlay == "corr" and corr is not None:
            col = "#f39c12" if corr[g] else "#bdc3c7"
        else:
            col = "#e74c3c" if y[g] else "#3498db"
        r, extra = (2.0, "") if g not in picked else (
            4.5, ' stroke="black" stroke-width="1.5"')
        out.append(f'<circle cx="{sc(xs[i,0],x0,x1):.1f}" cy="{sc(xs[i,1],y0,y1):.1f}"'
                   f' r="{r}" fill="{col}"{extra}/>')
    out.append("</svg>")
    return "".join(out)


@app.get("/")
def index():
    view = request.args.get("view", "spend")
    if view == "poison" and P is not None:
        s = request.args.get("s", "entropy")
        o = request.args.get("o", "sel")
        if s not in PSTRATS:
            s = "entropy"
        rows = [(k, float(P[f"corrfrac_{k}"]), float(P[f"auc_{k}"])) for k in PSTRATS]
        return render_template_string(
            POISON, ss=PSTRATS, s=s, o=o, rows=rows,
            ent_auc=float(P["auc_entropy"]), div_auc=float(P["auc_jepa-uniform"]),
            svg=scatter(P["xy"], P["y"], P[f"sel_{s}"], P["corr"], o)
            + "<p>ringed = selected for labeling</p>")
    if view == "map" and P is not None:
        r = request.args.get("r", "jepa-uniform")
        if r not in ("jepa-uniform", "jepa-guided"):
            r = "jepa-uniform"
        return render_template_string(
            MAP, r=r, svg=scatter(P["xy"], P["y"], P[f"sel_{r}"])
            + f"<p>k-center coverage on <b>{r}</b> (10% budget, phoneme)</p>")
    ds = request.args.get("d", "phoneme")
    b = float(request.args.get("b", 0.4))
    sel = request.args.get("s", "guided-mix")
    if ds not in DATASETS:
        ds = "phoneme"
    score = float(D[f"{ds}/auc_{sel}_{b}"])
    base = float(D[f"{ds}/auc_random_{b}"])
    return render_template_string(
        SPEND, ds=DATASETS, bs=BUDGETS, ss=STRATS, d=ds, b=b, sel=sel,
        score=score, base=base, svg=scatter(D[f"{ds}/xy"], D[f"{ds}/y"],
                                            D[f"{ds}/sel_{sel}_{b}"])
        + "<p>● class 1 ● class 0 (ringed = selected)</p>", abl=ABL)


if __name__ == "__main__":
    app.run(debug=True, port=5001)
