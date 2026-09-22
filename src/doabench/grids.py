"""Search grids.

Built from a point count, never ``np.arange`` with a float step: accumulated round-off in
``arange`` can drop or duplicate the endpoint (1800 or 1802 points instead of 1801).
"""

import numpy as np
from numpy.typing import NDArray


def uniform_grid(start: float, stop: float, step: float) -> NDArray[np.float64]:
    """Equivalent of MATLAB's ``start:step:stop`` for a step that divides the span."""
    n = round((stop - start) / step) + 1
    if not np.isclose(start + (n - 1) * step, stop):
        raise ValueError(f"step {step} does not divide [{start}, {stop}]")
    return np.linspace(start, stop, n)


def window(center: float, half_width: float, step: float) -> NDArray[np.float64]:
    """Local window ``center-half_width : step : center+half_width``."""
    return uniform_grid(center - half_width, center + half_width, step)
