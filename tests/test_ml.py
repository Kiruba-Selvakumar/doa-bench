import numpy as np
import pytest

from doabench.arrays import SensorArray
from doabench.estimators import ml_cost, ml_grid, ml_loop
from doabench.estimators.ml import _pair_costs
from doabench.grids import uniform_grid
from doabench.metrics import rmse


def test_cost_on_true_covariance_peaks_at_truth():
    arr = SensorArray.ula(4)
    theta = np.array([-20.0, 20.0])
    a = arr.steering(theta)
    r = a @ a.conj().T + 0.1 * np.eye(4)
    at_truth = ml_cost(r, a)
    for offset in [(-1, 0), (0, 1), (2, -2), (-5, 5)]:
        assert ml_cost(r, arr.steering(theta + np.array(offset))) < at_truth


def test_guard_rejects_identical_and_endfire_pairs():
    arr = SensorArray.ula(4)
    r = np.eye(4, dtype=complex)
    assert ml_cost(r, arr.steering([10.0, 10.0])) == -np.inf
    assert ml_cost(r, arr.steering([-90.0, 90.0])) == -np.inf


def test_vectorised_guard_rejects_endfire_pair_only():
    arr = SensorArray.ula(4)
    grid = uniform_grid(-90, 90, 1)
    steering = arr.steering(grid)
    cost = _pair_costs(np.eye(4, dtype=complex), steering, steering)
    upper = np.triu(np.ones_like(cost, dtype=bool), k=1)
    rejected = np.argwhere(upper & (cost == -np.inf))
    np.testing.assert_array_equal(rejected, [[0, grid.size - 1]])  # (-90°, +90°)


def test_vectorised_matches_loop(correlated, ml_grid_1deg):
    arr = correlated.array()
    for c in range(correlated.n_conditions):
        for t in (0, 57, 199):
            r = correlated.covariance(t, c)
            loop, _ = ml_loop(r, arr, 2, ml_grid_1deg)
            np.testing.assert_array_equal(ml_grid(r, arr, 2, ml_grid_1deg), loop)


def test_loop_handles_three_sources():
    arr = SensorArray.ula(6)
    theta = np.array([-30.0, 0.0, 30.0])
    a = arr.steering(theta)
    r = a @ a.conj().T + 0.1 * np.eye(6)
    est, _ = ml_loop(r, arr, 3, uniform_grid(-60, 60, 5), refine_step=0)
    np.testing.assert_array_equal(est, theta)


def test_vectorised_rejects_other_source_counts():
    with pytest.raises(NotImplementedError):
        ml_grid(np.eye(4, dtype=complex), SensorArray.ula(4), 3, uniform_grid(-90, 90, 1))


def test_refinement_removes_grid_floor(correlated, ml_grid_1deg):
    # Without refinement the 1° grid's quantisation floor (h/√12 ≈ 0.29°) makes ML look
    # worse than MUSIC (0.197°) at ρ = 0, inverting the headline result.
    arr = correlated.array()
    rs = [correlated.covariance(t, 0) for t in range(correlated.n_trials)]
    coarse = [ml_grid(r, arr, 2, ml_grid_1deg, refine_step=0) for r in rs]
    refined = [ml_grid(r, arr, 2, ml_grid_1deg) for r in rs]
    assert f"{rmse(coarse, correlated.theta_true):.3f}" == "0.395"
    assert f"{rmse(refined, correlated.theta_true):.3f}" == "0.198"
