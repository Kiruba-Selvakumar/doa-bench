import numpy as np
import pytest

from doabench.io import DOADataset


def test_uint8_metadata_is_converted_before_arithmetic(uncorrelated):
    # This file stores numTrials as uint8; 2 * uint8(200) wraps to 144 in numpy.
    assert type(uncorrelated.n_trials) is int
    assert 2 * uncorrelated.n_trials == 400
    assert type(uncorrelated.seed) is int


@pytest.mark.parametrize("name", ["correlated", "uncorrelated"])
def test_both_files_load_as_4d(name, request):
    ds = request.getfixturevalue(name)
    assert ds.snapshots.ndim == 4
    assert (ds.n_sensors, ds.n_snapshots, ds.n_trials) == (4, 100, 200)
    np.testing.assert_array_equal(ds.theta_true, [-20.3, 19.7])
    assert ds.d_over_lambda == 0.5
    assert ds.sigma2 == pytest.approx(0.1)


def test_condition_axes(correlated, uncorrelated):
    np.testing.assert_array_equal(correlated.rho, [0, 0.5, 0.9, 0.95, 0.99])
    assert correlated.seed == 20260904
    np.testing.assert_array_equal(uncorrelated.rho, [0.0])
    assert uncorrelated.seed == 20260905


def test_covariance_is_hermitian_psd(correlated):
    r = correlated.covariance(0, 4)
    np.testing.assert_allclose(r, r.conj().T, atol=1e-14)
    assert np.all(np.linalg.eigvalsh(r) > 0)


def test_dataset_rejects_inconsistent_shapes():
    x = np.zeros((4, 10, 3, 2), dtype=complex)
    with pytest.raises(ValueError, match="rho"):
        DOADataset(x, np.array([0.0]), np.array([-10.0, 10.0]), 0.5, 10.0, 0.1)
    with pytest.raises(ValueError, match="ascending"):
        DOADataset(x, np.array([0.0, 0.5]), np.array([10.0, -10.0]), 0.5, 10.0, 0.1)
