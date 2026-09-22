import numpy as np
import pytest

from doabench.grids import uniform_grid, window


@pytest.mark.parametrize(("step", "count"), [(0.1, 1801), (0.5, 361), (1, 181), (2, 91)])
def test_uniform_grid_has_matlab_point_count(step, count):
    g = uniform_grid(-90, 90, step)
    assert g.size == count
    assert g[0] == -90
    assert g[-1] == 90


def test_uniform_grid_rejects_step_that_does_not_divide_span():
    with pytest.raises(ValueError, match="does not divide"):
        uniform_grid(0, 1, 0.3)


def test_window_is_centred():
    w = window(-20.0, 1.0, 0.1)
    assert w.size == 21
    np.testing.assert_allclose(w[10], -20.0)
