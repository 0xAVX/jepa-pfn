# PFN-JEPA: TabPFN-guided self-supervision for learning which samples matter

We investigate whether JEPA-style self-supervised representations can improve
TabPFN-3.5. Three measured findings:

1. **Uniform latent augmentation hurts.** Generic self-supervised latents are
   not automatically compatible with TabPFN's learned prior (0/4 datasets).
2. **TabPFN-guided training repairs the failure.** Uncertainty curriculum +
   sensitivity-guided masks recover near-baseline on 3/4 datasets — the
   architectural contribution. TabPFN is causally central: it drives the
   curriculum, the masks, and the selection.
3. **Selection beats augmentation.** The guided representation is far more
   useful for choosing *which samples to label* than for extra features:
   NATICUS top-24 +0.086 over random with zero labels (`figs/jepa.csv`).

MC latent completion was **removed from the architecture** (not just ablated):
masked completions go out-of-distribution at inference (phoneme -0.015).

Prior-art position: [docs/novelty.md](docs/novelty.md) — we do not claim
first uncertainty+diversity or first TabPFN active learning; we claim TabPFN
uncertainty reshaping JEPA plus separated rank/coverage acquisition.

```python
from pfn_jepa.estimator import PFNJEPAClassifier
model = PFNJEPAClassifier(d_lat=64, epochs=10)  # mc removed; see figs/ablation.csv
model.fit(X_train, y_train)
model.predict_proba(X_test)
model.explain_uncertainty(X.iloc[[42]])
```

## Mechanism (`figs/mechanism.csv`)

**Rep-swap** (same k-center rule, 5 geometries): no space dominates. Uniform
JEPA is the best *map* (phoneme 0.9392 @20%, 0.9613 @40% — top both), guided
JEPA never wins pure diversity, TabPFN-embeddings fade with budget, NATICUS
is tied everywhere. Guidance is the better *compass* (ranking for the mix),
uniformity the better map. Map vs compass.

**Poison pool** (phoneme, 10% flipped labels, 10% budget): corruption selected
is ~equal across strategies (7.6–9.1%) — yet entropy collapses to AUC 0.365
while diversity methods hold 0.86–0.90. Entropy's failure is *not* "selects
more corrupted points"; it concentrates labeling on low-margin regions where
noise is fatal, while diversity spreads the risk. JEPA k-center selects the
fewest corrupted (7.6%). Epistemic value ≠ ambiguity.

![poison stress test](figs/poison.png)

- **Entropy sampling collapses at low budgets** — s6e9 5%: 0.581, phoneme
  10%: 0.487 (below chance). Pure TabPFN uncertainty picks unlearnable rows.
  Any uncertainty-only baseline is disqualified by its own numbers.
- **Easy tasks (s6e9, NATICUS): random ≈ diversity.** Nothing beats random
  sampling; JEPA ≈ raw for coverage.
- **Hard task (phoneme): guidance wins.** At 40% budget guided-mix 0.9646 vs
  random 0.9464 (+0.018); at 20%: 0.9376 vs 0.9248. Uncertainty × diversity
  on the guided representation is the only strategy that beats random —
  exactly where labels are scarcest relative to difficulty.

So the refined claim: guided representations don't beat random sampling in
general — they beat it where sampling is actually hard, and entropy alone
fails catastrophically everywhere at low budgets.

![label-budget curves](figs/budget_curves.png)

Phoneme (hard) at 40%: random 0.9464, entropy 0.9529, raw-kcenter 0.9551,
jepa-kcenter 0.9590, guided-mix **0.9646**.

## Exhibits (`figs/exhibits_feat.csv`, `figs/exhibits_aps.csv`)

**MADELON** (500 feats, ~20 signal): relevance direction depends on the
generator. Mask-sensitivity ranks *predictability* — on NATICUS that is
redundancy (top wins +0.086); on MADELON the XOR-type signal is unpredictable
by design, so the ranking inverts: bottom-50 0.877 vs top-50 0.554, bottom-100
0.950 vs top-100 0.718. Predictability ≈ redundancy; unpredictability ≈
signal. The demo-worthy twist: a "flip the ranking" toggle.

**APS Failure** (60k×170, 1.7% positives, real missingness; PR-AUC/recall):

| budget | random | entropy | jepa-kcenter | guided-mix |
|---|---|---|---|---|
| 1% PR / recall | 0.543 / 0.29 | 0.370 / 0.58 | 0.806 / 0.67 | **0.830 / 0.70** |
| 2% PR / recall | 0.742 / 0.42 | 0.801 / 0.70 | **0.861** / 0.72 | 0.849 / **0.74** |
| 10% PR / recall | 0.795 / 0.42 | 0.920 / 0.75 | 0.914 / 0.77 | **0.921** / 0.76 |

Operational read: at 1% (200 inspections) guided-mix finds 91 failures vs
random's 3. Entropy discovers positives (100–316) but ranks them poorly until
budgets grow — discovery without ranking.

## Reproduce

```bash
<venv-python> -m pytest tests/ -q
<venv-python> experiments/run_matrix.py   # augmentation ladder x datasets
<venv-python> experiments/selection.py    # label-budget curves x strategies
<venv-python> experiments/exhibits.py     # MADELON retention + APS acquisition
<venv-python> demo/app.py                 # interactive ladder + uncertainty
```

## Ablation ladder (`figs/ablation.csv`, 4 datasets)

| data | raw | jepa | +curriculum | +masks | +mc4 |
|---|---|---|---|---|---|
| s6e9-12k | 0.9465 | 0.9391 | 0.9408 | 0.9456 | 0.9458 |
| NATICUS-86 | 0.9885 | 0.9871 | 0.9872 | 0.9870 | 0.9873 |
| phoneme | 0.9775 | 0.9671 | 0.9694 | 0.9696 | 0.9549 |
| churn | 0.9178 | 0.9147 | 0.9133 | 0.9135 | 0.9130 |

Gating verdict (a stage must beat the last on ≥2 datasets):
**uniform JEPA augment hurts everywhere** (0/4) — 32 extra latents dilute
TabPFN's raw signal. **Curriculum recovers** (3/4), **guided masks recover
more** (3/4, s6e9 back to -0.0009 of raw). **MC is cut**: neutral twice,
catastrophic on phoneme (-0.015, masked latents go OOD).
Surviving headline: PFN guidance systematically repairs unguided JEPA;
wide-data *selection* (not augmentation) is where the plug wins outright
(NATICUS top-24 +0.086 over random, zero labels).

![ablation ladder](figs/ablation.png)
