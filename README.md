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

## Ablation ladder (target README figure)

TabPFN raw → JEPA→TabPFN → +curriculum → +guided masks → +diagnostics → +MC.
Each stage must beat the last on ≥2 datasets or it gets cut and reported.
