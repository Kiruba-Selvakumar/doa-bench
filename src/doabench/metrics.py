"""Error metrics over Monte Carlo trials.

``estimates`` has shape (n_trials, M) with each row ascending; ``theta_true`` is ascending,
so sources are paired by rank. A row of NaN marks a trial where the estimator declined to
answer (e.g. MUSIC found fewer than M peaks).
"""

import numpy as np
from numpy.typing import ArrayLike


def rmse(estimates: ArrayLike, theta_true: ArrayLike) -> float:
    """Root-mean-square error in degrees over all non-NaN estimates."""
    err = np.asarray(estimates, dtype=float) - np.asarray(theta_true, dtype=float)
    return float(np.sqrt(np.nanmean(err**2)))


def resolution_tolerance(theta_true: ArrayLike) -> float:
    """Half the smallest separation: each estimate is nearer its own source than any other."""
    return float(np.min(np.diff(np.sort(np.asarray(theta_true, dtype=float)))) / 2)


def success_rate(estimates: ArrayLike, theta_true: ArrayLike, tol: float | None = None) -> float:
    """Fraction of trials in which every source is recovered to within ``tol`` degrees.

    ``tol`` defaults to ``resolution_tolerance(theta_true)``. NaN estimates count as failures.
    """
    theta = np.asarray(theta_true, dtype=float)
    tol = resolution_tolerance(theta) if tol is None else tol
    err = np.abs(np.asarray(estimates, dtype=float) - theta)
    with np.errstate(invalid="ignore"):
        ok = np.all(err <= tol, axis=1)
    return float(np.mean(ok))


def failure_count(estimates: ArrayLike) -> int:
    """Number of trials returning no estimate (any NaN in the row)."""
    return int(np.sum(np.any(np.isnan(np.asarray(estimates, dtype=float)), axis=1)))
