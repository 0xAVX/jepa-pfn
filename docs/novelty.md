# Novelty audit: where JEPAFN sits

Literature cut-off: September 2026 (Papers With Code catalog + arXiv).

## 1. Closest prior work

| Method | TabPFN | SSL repr. | Uncertainty reshapes repr. learning | Diversity for acquisition | Separate rank/coverage spaces |
|---|:---:|:---:|:---:|:---:|:---:|
| T-JEPA (2410.05016) † | — | ✓ | — | — | — |
| Tab-AICL (Treerath & Pittorino, Mar 2026, arXiv:2603.27385) | ✓ | — | — | ✓ | — |
| TCM / SSL+AL schedule (Doucet 2024) ¶ | — | ✓ | — | ✓ | — |
| Less-is-More tabular SSL (Springer 2026) ‡ | — | ✓ | — | ✓ | — |
| UA-Tab (SciDirect 2026) § | — | ✓ | ✓ | — | — |
| UnDi (SciDirect 2026, vision-language) ¶ | — | ✓ | — | ✓ | — |
| **JEPAFN (this work)** | **✓** | **✓** | **✓** | **✓** | **✓** |

† T-JEPA learns tabular JEPA representations but is not a TabPFN-guided
active-learning loop (official repo: jose-melo/t-jepa).
‡ Less-is-More studies uncertainty/diversity for selecting data used in
tabular SSL, but its selection criteria are explicitly independent of the SSL
model (no embeddings, gradients, losses, augmentations) — near-inverse of our
feedback loop.
§ UA-Tab is the closest hit for uncertainty-aware tabular SSL, but its
uncertainty is feature-level and internal to the representation learner — not
downstream TabPFN predictive uncertainty steering JEPA plus active acquisition.
¶ TCM-style work schedules diversity→uncertainty rather than model-guided
representation feedback. UnDi clusters in a fixed pretrained space; it never
reshapes the representation with downstream uncertainty.

## 2. What is already known

- Pure uncertainty sampling can over-focus on boundaries/outliers; diversity
  is the standard fix (active-learning literature, e.g. PMC4144157).
- Tab-AICL already combines TabPFN entropy shortlists with k-means diversity.
- Self-supervised tabular representations (T-JEPA) can identify
  downstream-relevant structure without labels.

## 3. What JEPAFN changes

JEPAFN couples a tabular foundation model with self-supervision in a feedback
loop: TabPFN predictive uncertainty guides JEPA training (curriculum + masks),
while acquisition deliberately separates task relevance from coverage — guided
representations rank, unguided JEPA geometry covers. Central empirical point:
**the same representation is harmful as a feature (0/4 augmentation) and
useful as a geometry (+0.086 feature selection, guided-mix wins).**

## 4. Claims we do / do not make

We do NOT claim: first uncertainty+diversity AL (Tab-AICL's Margin/Coreset/
Hybrid/Proxy-Hybrid rules already combine TabPFN uncertainty with diversity
on fixed spaces, evaluated as cold-start AULC on 20 benchmarks);
first TabPFN active learning method;
first self-supervised AL; first uncertainty-aware tabular SSL.

We DO contribute: TabPFN predictive uncertainty as feedback into JEPA
training; empirical separation of guided ranking vs unguided coverage;
their recombination for label allocation; mechanistic evidence for entropy
failure under matched corruption rates; evidence that JEPA representations
help selection even when direct augmentation fails.
