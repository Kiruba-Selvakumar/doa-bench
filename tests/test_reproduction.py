"""The Python port reproduces the original MATLAB implementation, digit for digit.

Every expected value below was printed by the MATLAB implementation (R2025b) on the datasets
in ``data/reference/``. Values are compared as formatted strings at the precision MATLAB
printed them, so a pass means the console output would be identical.
"""

from math import comb

import numpy as np
import pytest

from doabench.estimators import ml_grid, ml_loop, music
from doabench.metrics import failure_count, rmse, success_rate

# rho: (RMSE MUSIC, RMSE ML, success MUSIC %, success ML %, MUSIC no-peak trials)
CORRELATION_SWEEP = {
    0.00: ("0.197", "0.198", "100.0", "100.0", 0),
    0.50: ("0.237", "0.227", "100.0", "100.0", 0),
    0.90: ("0.494", "0.319", "100.0", "100.0", 0),
    0.95: ("0.885", "0.363", "100.0", "100.0", 0),
    0.99: ("6.197", "0.389", "97.5", "100.0", 1),
}
UNCORRELATED_BASELINE = ("0.197", "0.195", "100.0", "100.0", 0)

# Sorted eigenvalues of R̂ for trial 1 of each rho, and the lambda2/lambda3 gap.
TRIAL1_EIGENVALUES = {
    0.00: ("4.2308", "2.1412", "0.1234", "0.0763", "17.35"),
    0.50: ("6.4719", "1.3662", "0.1217", "0.0805", "11.23"),
    0.90: ("10.7692", "0.3615", "0.1254", "0.0846", "2.88"),
    0.95: ("10.5331", "0.2431", "0.1211", "0.0814", "2.01"),
    0.99: ("8.5590", "0.1276", "0.1146", "0.0983", "1.11"),
}


def sweep(ds, music_grid, ml_grid_1deg):
    arr, m = ds.array(), ds.n_sources
    out = {}
    for c in range(ds.n_conditions):
        rs = [ds.covariance(t, c) for t in range(ds.n_trials)]
        out[round(float(ds.rho[c]), 2)] = (
            np.array([music(r, arr, m, music_grid).theta for r in rs]),
            np.array([ml_grid(r, arr, m, ml_grid_1deg) for r in rs]),
        )
    return out


@pytest.fixture(scope="module")
def correlated_sweep(correlated, music_grid, ml_grid_1deg):
    return sweep(correlated, music_grid, ml_grid_1deg)


def row(est_music, est_ml, truth):
    return (
        f"{rmse(est_music, truth):.3f}",
        f"{rmse(est_ml, truth):.3f}",
        f"{100 * success_rate(est_music, truth):.1f}",
        f"{100 * success_rate(est_ml, truth):.1f}",
        failure_count(est_music),
    )


@pytest.mark.parametrize("rho", sorted(CORRELATION_SWEEP))
def test_correlation_sweep(rho, correlated_sweep, correlated):
    assert row(*correlated_sweep[rho], correlated.theta_true) == CORRELATION_SWEEP[rho]


def test_headline_ratios(correlated_sweep, correlated):
    truth = correlated.theta_true
    r = {rho: [rmse(e, truth) for e in est] for rho, est in correlated_sweep.items()}
    assert f"{r[0.99][0] / r[0.00][0]:.1f}" == "31.4"  # MUSIC degradation
    assert f"{r[0.99][1] / r[0.00][1]:.1f}" == "2.0"  # ML degradation
    assert f"{r[0.99][0] / r[0.99][1]:.1f}" == "15.9"  # MUSIC / ML at rho = 0.99


def test_uncorrelated_baseline(uncorrelated, music_grid, ml_grid_1deg):
    (est,) = sweep(uncorrelated, music_grid, ml_grid_1deg).values()
    assert row(*est, uncorrelated.theta_true) == UNCORRELATED_BASELINE


@pytest.mark.parametrize("c", range(5))
def test_eigenvalue_collapse(c, correlated):
    ev = np.sort(np.linalg.eigvalsh(correlated.covariance(0, c)))[::-1]
    got = (*(f"{e:.4f}" for e in ev), f"{ev[1] / ev[2]:.2f}")
    assert got == TRIAL1_EIGENVALUES[round(float(correlated.rho[c]), 2)]


def test_single_trial_estimates(correlated, music_grid, ml_grid_1deg):
    r, arr = correlated.covariance(0, 0), correlated.array()
    est_ml, n_eval = ml_loop(r, arr, 2, ml_grid_1deg)
    assert [f"{t:.3f}" for t in music(r, arr, 2, music_grid).theta] == ["-20.600", "19.800"]
    assert [f"{t:.3f}" for t in est_ml] == ["-20.600", "19.800"]
    # 16290 pairs, 1 rejected by the determinant guard (-90°, +90°), plus 21 x 21 refinement.
    assert n_eval == 16730 == comb(181, 2) - 1 + 21 * 21


def test_coherent_failure_pseudospectrum(correlated, music_grid):
    arr = correlated.array()
    benign = music(correlated.covariance(0, 0), arr, 2, music_grid)
    failed = music(correlated.covariance(0, 4), arr, 2, music_grid)
    assert np.all(np.isnan(failed.theta))  # no two distinct peaks at rho = 0.99
    assert f"{10 * np.log10(benign.spectrum.max()):.1f}" == "28.1"
    assert f"{10 * np.log10(failed.spectrum.max()):.1f}" == "0.9"
