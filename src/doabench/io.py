"""Loading the reference ``.mat`` datasets.

MATLAB's ``-v7`` format compacts integer-valued doubles into the smallest lossless integer
class. MATLAB converts them back on load; scipy does not. ``numTrials = 200`` can therefore
arrive as ``uint8``, and ``2 * numTrials`` then silently wraps to 144. Every scalar is
converted to a plain Python ``int``/``float`` here, before any arithmetic touches it.
"""

from dataclasses import dataclass
from os import PathLike

import numpy as np
import scipy.io as sio
from numpy.typing import NDArray

from doabench.arrays import SensorArray
from doabench.covariance import sample_covariance


@dataclass(frozen=True)
class DOADataset:
    """Snapshots for several trials of several source-correlation conditions."""

    snapshots: NDArray[np.complex128]  # (N, L, n_trials, n_conditions)
    rho: NDArray[np.float64]  # (n_conditions,)
    theta_true: NDArray[np.float64]  # (M,), degrees, ascending
    d_over_lambda: float
    snr_db: float
    sigma2: float
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.snapshots.ndim != 4:
            raise ValueError(f"snapshots must be 4-D, got shape {self.snapshots.shape}")
        if self.snapshots.shape[3] != self.rho.size:
            raise ValueError(
                f"{self.snapshots.shape[3]} conditions in snapshots but {self.rho.size} rho values"
            )
        if np.any(np.diff(self.theta_true) <= 0):
            raise ValueError("theta_true must be strictly ascending")

    @property
    def n_sensors(self) -> int:
        return self.snapshots.shape[0]

    @property
    def n_snapshots(self) -> int:
        return self.snapshots.shape[1]

    @property
    def n_trials(self) -> int:
        return self.snapshots.shape[2]

    @property
    def n_conditions(self) -> int:
        return self.snapshots.shape[3]

    @property
    def n_sources(self) -> int:
        return self.theta_true.size

    def array(self) -> SensorArray:
        return SensorArray.ula(self.n_sensors, self.d_over_lambda)

    def covariance(self, trial: int, condition: int) -> NDArray[np.complex128]:
        return sample_covariance(self.snapshots[:, :, trial, condition])


def load_reference_mat(path: str | PathLike[str]) -> DOADataset:
    """Load a dataset written by the original MATLAB generator (``-v7`` .mat file)."""
    m = sio.loadmat(path)

    def scalar_int(key: str) -> int:
        return int(m[key].item())

    def scalar_float(key: str) -> float:
        return float(m[key].item())

    x = np.asarray(m["X"], dtype=np.complex128)
    if x.ndim == 3:  # single-condition file: MATLAB drops the trailing singleton
        x = x[:, :, :, np.newaxis]

    ds = DOADataset(
        snapshots=x,
        rho=np.atleast_1d(m["rho"].squeeze()).astype(np.float64),
        theta_true=np.atleast_1d(m["thetaTrue"].squeeze()).astype(np.float64),
        d_over_lambda=scalar_float("dOverLambda"),
        snr_db=scalar_float("snrDb"),
        sigma2=scalar_float("sigma2"),
        seed=scalar_int("seed") if "seed" in m else None,
    )

    # The stored metadata must agree with the array it describes.
    stored = {k: scalar_int(k) for k in ("N", "L", "numTrials")}
    actual = {"N": ds.n_sensors, "L": ds.n_snapshots, "numTrials": ds.n_trials}
    if stored != actual:
        raise ValueError(f"{path}: metadata {stored} disagrees with X shape {actual}")
    return ds
