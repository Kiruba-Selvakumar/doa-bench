"""MUSIC (MUltiple SIgnal Classification).

The N-M eigenvectors of R̂ with the smallest eigenvalues span the noise subspace Eₙ, which
is orthogonal to every true steering vector. The pseudospectrum

    P(θ) = 1 / ‖Eₙᴴ a(θ)‖²

therefore peaks at the true DOAs. Only the peak *locations* are estimates; the heights
carry no physical meaning.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from doabench.arrays import SensorArray


@dataclass(frozen=True)
class MusicResult:
    theta: NDArray[np.float64]  # (M,) ascending, or all-NaN if fewer than M peaks
    spectrum: NDArray[np.float64]  # (G,) linear scale
    grid: NDArray[np.float64]  # (G,) degrees


def music_spectrum(
    covariance: NDArray[np.complex128],
    array: SensorArray,
    n_sources: int,
    grid: NDArray[np.float64],
) -> NDArray[np.float64]:
    """MUSIC pseudospectrum on ``grid`` (linear scale)."""
    _, vecs = np.linalg.eigh(covariance)  # eigenvalues ascending
    noise = vecs[:, : array.n_sensors - n_sources]  # N x (N-M)
    residual = np.sum(np.abs(noise.conj().T @ array.steering(grid)) ** 2, axis=0)
    return 1.0 / residual


def local_maxima(
    values: NDArray[np.float64], grid: NDArray[np.float64], count: int
) -> NDArray[np.float64]:
    """Locations of the ``count`` tallest interior local maxima, ascending.

    The ``count`` largest *values* are the wrong answer: they sit on the flank of the same
    tall peak and would report one source twice. Returns all-NaN when fewer than ``count``
    maxima exist, e.g. when two peaks merge or a weak source sinks into the noise floor, so
    the caller can count a resolution failure instead of reporting a wrong angle.
    """
    v = np.asarray(values)
    interior = (v[1:-1] > v[:-2]) & (v[1:-1] >= v[2:])
    peaks = np.flatnonzero(interior) + 1
    if peaks.size < count:
        return np.full(count, np.nan)
    tallest = peaks[np.argsort(-v[peaks], kind="stable")[:count]]
    return np.sort(grid[tallest])


def music(
    covariance: NDArray[np.complex128],
    array: SensorArray,
    n_sources: int,
    grid: NDArray[np.float64],
) -> MusicResult:
    """MUSIC DOA estimates for a known number of sources."""
    spectrum = music_spectrum(covariance, array, n_sources, grid)
    return MusicResult(local_maxima(spectrum, grid, n_sources), spectrum, grid)
