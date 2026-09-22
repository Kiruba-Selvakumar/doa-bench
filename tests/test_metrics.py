import numpy as np
import pytest

from doabench.metrics import failure_count, resolution_tolerance, rmse, success_rate

TRUTH = np.array([-20.0, 20.0])


def test_rmse_pools_all_sources():
    est = np.array([[-19.0, 21.0], [-20.0, 17.0]])
    assert rmse(est, TRUTH) == pytest.approx(np.sqrt((1 + 1 + 0 + 9) / 4))


def test_rmse_skips_nan_trials():
    est = np.array([[-19.0, 21.0], [np.nan, np.nan]])
    assert rmse(est, TRUTH) == pytest.approx(1.0)


def test_rmse_can_mislead():
    # 195 trials at 0.1°, 5 trials 40° off -> RMSE ≈ 6.3°, which describes neither behaviour.
    est = np.tile(TRUTH + 0.1, (200, 1))
    est[:5] = TRUTH + 40
    assert rmse(est, TRUTH) == pytest.approx(6.3, abs=0.05)


def test_resolution_tolerance_is_half_the_separation():
    assert resolution_tolerance([-20.3, 19.7]) == pytest.approx(20.0)


def test_success_requires_every_source_and_counts_nan_as_failure():
    est = np.array(
        [
            [-19.0, 21.0],  # both within 20°
            [-19.0, -1.0],  # second source 21° off
            [np.nan, np.nan],  # no answer
            [-39.9, 39.9],  # both just inside
        ]
    )
    assert success_rate(est, TRUTH) == pytest.approx(0.5)
    assert failure_count(est) == 1
