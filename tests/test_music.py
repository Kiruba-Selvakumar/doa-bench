import numpy as np
import pytest

from doabench.arrays import SensorArray
from doabench.estimators import local_maxima, music


@pytest.fixture
def worked_example():
    """True covariance for N=6, sources at -35° and +10°, unit powers, σ² = 0.2."""
    arr = SensorArray.ula(6)
    theta = np.array([-35.0, 10.0])
    a = arr.steering(theta)
    return arr, theta, a @ a.conj().T + 0.2 * np.eye(6)


def test_worked_example_eigenvalues(worked_example):
    _, _, r = worked_example
    ev = np.sort(np.linalg.eigvalsh(r))[::-1]
    np.testing.assert_allclose(ev[:2], [6.946, 5.454], atol=5e-4)
    np.testing.assert_allclose(ev[2:], 0.2, atol=1e-12)  # exactly N-M copies of σ²


def test_worked_example_orthogonality_residuals(worked_example):
    arr, theta, r = worked_example
    _, vecs = np.linalg.eigh(r)
    noise = vecs[:, :4]

    def residual(t):
        return float(np.sum(np.abs(noise.conj().T @ arr.steering(t)) ** 2))

    for t in theta:
        assert residual(t) < 1e-25  # zero to machine precision
    assert residual(0.0) == pytest.approx(3.36, abs=5e-3)
    assert residual(25.0) == pytest.approx(5.41, abs=5e-3)


def test_music_on_true_covariance_returns_true_doas(worked_example, music_grid):
    arr, theta, r = worked_example
    np.testing.assert_allclose(music(r, arr, 2, music_grid).theta, theta, atol=1e-9)


def test_local_maxima_are_peaks_not_largest_values():
    grid = np.arange(11.0)
    #                  one tall broad peak around 3, one small peak at 8
    values = np.array([0, 5, 9, 10, 9, 5, 1, 2, 3, 2, 1], dtype=float)
    # The two largest values (10 and 9) are both on the first peak.
    np.testing.assert_array_equal(local_maxima(values, grid, 2), [3.0, 8.0])


def test_local_maxima_returns_nan_when_too_few_peaks():
    values = np.array([0, 1, 2, 3, 2, 1, 0], dtype=float)
    assert np.all(np.isnan(local_maxima(values, np.arange(7.0), 2)))


def test_local_maxima_ignores_endpoints():
    # A monotone spectrum has no interior maximum even though its edge is the largest value.
    assert np.all(np.isnan(local_maxima(np.arange(5.0), np.arange(5.0), 1)))
