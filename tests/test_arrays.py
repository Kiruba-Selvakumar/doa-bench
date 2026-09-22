import numpy as np
import pytest

from doabench.arrays import SensorArray


def reference_steering(theta_deg, n, d_over_lambda):
    """The MATLAB expression: exp(1i*2*pi*dOverLambda*(0:N-1)'*sind(theta))."""
    n_idx = np.arange(n)[:, None]
    return np.exp(1j * 2 * np.pi * d_over_lambda * n_idx * np.sin(np.deg2rad(theta_deg))[None, :])


@pytest.mark.parametrize(("n", "spacing"), [(4, 0.5), (8, 0.5), (6, 0.25)])
def test_ula_matches_reference_convention(n, spacing):
    grid = np.linspace(-90, 90, 361)
    np.testing.assert_allclose(
        SensorArray.ula(n, spacing).steering(grid),
        reference_steering(grid, n, spacing),
        atol=1e-12,
    )


def test_steering_has_unit_modulus_and_first_sensor_reference():
    a = SensorArray.ula(5).steering([-40.0, 0.0, 33.3])
    np.testing.assert_allclose(np.abs(a), 1.0)
    np.testing.assert_allclose(a[0], 1.0)


def test_broadside_is_in_phase():
    np.testing.assert_allclose(SensorArray.ula(4).steering(0.0), 1.0)


def test_half_wavelength_ula_cannot_distinguish_the_two_endfire_directions():
    # The root cause of the ML determinant trap: a(-90°) and a(+90°) coincide.
    a = SensorArray.ula(4).steering([-90.0, 90.0])
    np.testing.assert_allclose(a[:, 0], a[:, 1], atol=1e-12)


def test_scalar_angle_gives_single_column():
    assert SensorArray.ula(4).steering(12.5).shape == (4, 1)


def test_positions_are_validated_and_immutable():
    with pytest.raises(ValueError, match="shape"):
        SensorArray(np.zeros((4, 2)))
    arr = SensorArray.ula(4)
    with pytest.raises(ValueError):
        arr.positions[0, 0] = 1.0
