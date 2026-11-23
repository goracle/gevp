# Three-slice GEVP

What can you extract from three time slices of GEVP two-point correlator data? Can you bound the energy from both sides? Can you beat the ALPHA Collaboration result?

This repo collects the source papers, the notes written to answer those questions, and the scripts that check the claims numerically.

> **Status:** working notes, not peer reviewed. The notes were written with GPT and Claude assistance. All numerics are **noise-free, real-symmetric toy models** with fixed seeds, except one crude noise test in `three_slice_checks.py` (T7). Nothing has been tested on real lattice correlators.

## Short version of the results

- **No improvement over ALPHA from three slices alone.** For the one-contaminant model the data fix the energies only up to a one-parameter family of exact fits (Jacobian deficiency 1 for every N). This also explains why the 2019 toy fit had inconsistent minima.
- **Upper bound:** the GEVP on the later pair (C(t+a), C(t)) is a Rayleigh–Ritz estimate, `E_n <= E_mov`. For the ground state, `E_1 <= E_mov <= E_ALPHA`. There is no ordering for interior levels.
- **Lower bound with a threshold:** if every unresolved level lies above `E_thr`, then with `s = exp(-a*E_thr)` the pencil `(m2 - s*m1) v = lambda (m1 - s*m0) v` gives `lambda_n(0) <= tau_n <= lambda_n(s)` (Theorem 1 in the notes). This is the classical Lehmann–Maehly bound written in GEVP form, not a new result.
- **Jensen defect:** `B - A^p` is positive semidefinite for `1 <= p <= 2` only. It fails for `p > 2` (808/1500 violations at `q = 2.5` in the checks).
- **A fourth slice** is what gives an identifiable one-contaminant model, with a large gain only for an isolated first contaminant.

## Layout

### Papers and talks (background)

| File | What it is |
|---|---|
| `0902_1265v2.pdf` | Blossier et al. (ALPHA), arXiv:0902.1265v2. Baseline: error in E_n is O(e^{-(E_{N+1}-E_n)t}) for t0 >= t/2. |
| `talk_gevp.pdf` | Hoying, "Three t GEVP", 24 Sep 2019, Lattice X Intensity Frontier Workshop, BNL. Poses the three-slice questions. Off-diagonal series is a sketch. |
| `pseudosort.pdf` | Hoying, "Sorting GEVP Eigenvalues Across Time Slices" (pseudo-eigenvalue level matching). Separate topic from the three-slice analysis. |

### Notes (read in this order)

| File | What it is |
|---|---|
| `three_slice_gevp_notes.pdf` | First working notes (Claude). Contains Prop. 1 (exactness), Theorem 1 (threshold bracket), the secular equation. **Sec. 6 is wrong** (the counting table and the W(s) rank-one root scan); it is withdrawn in the assessment. |
| `three_t_gevp_alpha_comparison_notes.pdf`, `.tex` | GPT notes: ALPHA vs moving-pair GEVP, operator-Jensen defect, one-extra-state completion M(c). |
| `three_t_gevp_defect_noise_notes.pdf` | GPT notes on noisy defect matrices: covariance-aware constrained fits and confidence-set bounds. Proposals, not tested here. |
| `three_slice_assessment.pdf` | Claude assessment of the first notes: verification table, the identifiability result, four-slice test, literature overlap, verdict. |
| `three_slice_gevp_reconciled_notes.pdf` | Merge of all of the above: notation dictionary, statement ledger with [src]/[hand]/[run] tags, conflicts resolved. **Start here.** |

### Scripts

All need `numpy` and `scipy`. Run from the repo root.

| File | Checks | Run |
|---|---|---|
| `three_slice_checks.py` | T1-T7: exactness, Theorem 1 bracket, unequal spacing, residual matrix W(s), Cauchy–Binet secular form, estimator comparison, noise amplification. Imported by `three_slice_checks2.py`. | `python three_slice_checks.py` |
| `three_slice_checks2.py` | T8-T11: Jacobian rank (3 vs 4 slices), the family of exact fits, what a fourth slice does. | `python three_slice_checks2.py` |
| `reconcile_checks.py` | T1-T6 for the reconciled notes: exactness, bracket, Jensen defect vs q, ALPHA vs moving-pair ordering, M(c) <-> s-family dictionary, Jacobian rank. | `python reconcile_checks.py` |

## Known gaps

- `three_slice_checks3.py` (Lehmann/Davis–Kahan comparison) and `three_slice_toy.py` are referenced in the notes but are not in this repo.
- The Lanczos/Prony and Davis–Kahan literature comparison is partial. See Sec. 7 of the assessment and Sec. 11 of the reconciled notes.
- Novelty of the arbitrary-spacing bracket (`h_s(tau) = (tau^q - s^q)/(tau - s)`) is unchecked. The `q = 2` case is the right-definite Lehmann bound.
- The off-diagonal perturbation series from the 2019 talk was never completed.

## Related literature

- Blossier et al. (ALPHA), arXiv:0902.1265.
- Abbott et al., "Filtered Rayleigh-Ritz is all you need", arXiv:2503.17357.
- Wagman, arXiv:2406.20009. Hackett and Wagman, arXiv:2412.04444. NPLQCD, arXiv:2601.22272.
- Beattie, "Variations of Ritz and Lehmann bounds", arXiv:math/9805029 (Lehmann–Maehly background).
