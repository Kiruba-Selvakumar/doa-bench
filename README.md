# doa-bench

Direction-of-arrival (DOA) estimators for sensor arrays, benchmarked against each other and,
from the next sprint on, against the Cramér–Rao bound.

The starting point is a two-estimator study: **MUSIC** (subspace method, cheap) against
**deterministic maximum likelihood** (grid search, O(G^M)) on a 4-element half-wavelength
uniform linear array with two sources. In benign conditions the two are equally accurate.
As the sources become correlated, as a direct path and its multipath reflection are, MUSIC
collapses while ML barely moves:

| source correlation ρ | RMSE MUSIC | RMSE ML | MUSIC success | ML success |
|---:|---:|---:|---:|---:|
| 0.00 | 0.197° | 0.198° | 100.0% | 100.0% |
| 0.50 | 0.237° | 0.227° | 100.0% | 100.0% |
| 0.90 | 0.494° | 0.319° | 100.0% | 100.0% |
| 0.95 | 0.885° | 0.363° | 100.0% | 100.0% |
| 0.99 | **6.197°** | **0.389°** | **97.5%** | 100.0% |

200 Monte Carlo trials per row; N = 4, L = 100 snapshots, SNR = 10 dB, true DOAs
[-20.3°, 19.7°].

The mechanism: the source covariance has eigenvalues 1 ± ρ, so as ρ → 1 the signal
subspace collapses toward rank 1, and MUSIC's signal/noise eigenvalue split disappears. In
the data, the sample gap λ₂/λ₃ falls from 17.35× to 1.11× across the sweep. ML never splits
R̂ into subspaces, so it is barely affected.

## Status

**Sprint 0: foundation.** This package is a Python port of an original MATLAB
implementation, and it reproduces that implementation digit for digit: every number above,
plus the eigenvalue tables, single-trial estimates, search-evaluation counts and
pseudospectrum peak heights. The MATLAB outputs are the expected values in
[`tests/test_reproduction.py`](tests/test_reproduction.py), and CI checks them on
Python 3.11–3.13. A first estimate from real multichannel recordings is in
[`spikes/`](spikes/).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync                                  # create the environment
uv run pytest                            # full suite, including the MATLAB reproduction
uv run pytest tests/test_reproduction.py # just the reproduction
```

```python
from doabench.io import load_reference_mat
from doabench.grids import uniform_grid
from doabench.estimators import music, ml_grid

ds = load_reference_mat("data/reference/doa_dataset_correlated.mat")
r = ds.covariance(trial=0, condition=4)  # rho = 0.99
music(r, ds.array(), 2, uniform_grid(-90, 90, 0.1)).theta  # -> [nan, nan]: no two peaks
ml_grid(r, ds.array(), 2, uniform_grid(-90, 90, 1))  # -> [-19.8, 19.6]
```

## Layout

| path | contents |
|---|---|
| `src/doabench/arrays.py` | sensor geometry from coordinates; steering vectors |
| `src/doabench/estimators/` | MUSIC; deterministic ML (explicit loop and vectorised) |
| `src/doabench/io.py` | dataset loader, safe against MATLAB's integer compaction |
| `src/doabench/metrics.py` | RMSE, resolution success rate |
| `data/reference/` | the two MATLAB-generated datasets, pinned by `SHA256SUMS` |
| `spikes/` | exploratory scripts: first DOA estimate from the LOCATA recordings |

## Two traps the tests pin down

- **The ML determinant guard.** With d = λ/2, the steering vectors for -90° and +90° are
  identical, so det(AᴴA) = 0 for that pair and round-off turns its cost into an arbitrary
  value that wins the search in many trials (RMSE about 60°). A guard based on angle
  separation misses it, since the two angles are 180° apart; a guard on det(AᴴA) catches it.
- **Grid quantisation.** A 1° ML grid has an RMS floor of 0.29°, above MUSIC's own error.
  Without local refinement, ML measures 0.395° against MUSIC's 0.197° at ρ = 0, which
  inverts the headline result.

## Roadmap

| sprint | theme |
|---|---|
| S0 | foundation: Python port, reproduction tests, CI ← *current* |
| S1 | Cramér–Rao bound and the analytic MUSIC variance |
| S2 | Bartlett, MVDR, Root-MUSIC, ESPRIT; model-order selection |
| S3 | coherent-source repair: forward–backward averaging, spatial smoothing, stochastic ML |
| S4 | wideband processing and simulated reverberation |
| S5 | real multichannel recordings (LOCATA) |
| S6 | operating-envelope maps: SNR × separation × snapshots × coherence |
| S7–S8 | model-based deep learning (deep-unfolded SBL) and a generalisation audit |
