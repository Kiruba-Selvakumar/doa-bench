"""Sample covariance."""

import numpy as np
from numpy.typing import NDArray


def sample_covariance(snapshots: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """R̂ = X Xᴴ / L for an N x L snapshot matrix X."""
    x = np.asarray(snapshots)
    if x.ndim != 2:
        raise ValueError(f"snapshots must be N x L, got shape {x.shape}")
    return (x @ x.conj().T) / x.shape[1]
