# PFN-JEPA

**Teaching a tabular world model what TabPFN doesn't know.** A closed loop:
TabPFN-3.5 identifies uncertain rows and fragile features (cross-fit, never
in-sample) → a JEPA self-supervised encoder trains harder exactly there
(uncertainty curriculum + sensitivity-guided masks, no labels) → latents,
recon error, completion variance and group errors feed back into TabPFN-3.5
→ optional MC latent-completion ensemble.

```python
from pfn_jepa.estimator import PFNJEPAClassifier
model = PFNJEPAClassifier(d_lat=64, epochs=10, mc_samples=8)
model.fit(X_train, y_train)
model.predict_proba(X_test)
model.explain_uncertainty(X.iloc[[42]])
```

## Reproduce

```bash
<venv-python> -m pytest tests/ -q
<venv-python> experiments/run_matrix.py   # ablation ladder x datasets
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
