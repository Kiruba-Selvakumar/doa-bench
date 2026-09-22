"""Sensor-array geometry and steering vectors.

Sensor positions are Cartesian coordinates in wavelengths, so one code path serves the
uniform linear array used today and arbitrary microphone geometries later. For the 1-D
problems here the array lies in the x-y plane and θ is measured from broadside (+y axis),
toward +x. For a ULA along the x axis this reproduces the reference convention

    a_n(θ) = exp(j·2π·(d/λ)·n·sin θ),   n = 0, …, N-1.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class SensorArray:
    """An array of isotropic sensors at fixed positions (in wavelengths)."""

    positions: NDArray[np.float64]  # shape (N, 3)

    def __post_init__(self) -> None:
        p = np.array(self.positions, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {p.shape}")
        p.setflags(write=False)
        object.__setattr__(self, "positions", p)

    @classmethod
    def ula(cls, n_sensors: int, spacing: float = 0.5) -> "SensorArray":
        """Uniform linear array along the x axis, first sensor at the origin."""
        positions = np.zeros((n_sensors, 3))
        positions[:, 0] = spacing * np.arange(n_sensors)
        return cls(positions)

    @property
    def n_sensors(self) -> int:
        return self.positions.shape[0]

    def steering(self, theta_deg: ArrayLike) -> NDArray[np.complex128]:
        """Steering matrix for broadside angles in degrees, shape (N, G)."""
        theta = np.deg2rad(np.atleast_1d(np.asarray(theta_deg, dtype=np.float64)))
        u = np.stack([np.sin(theta), np.cos(theta), np.zeros_like(theta)])  # (3, G)
        return np.exp(2j * np.pi * (self.positions @ u))
