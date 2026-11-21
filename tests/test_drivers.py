import numpy as np

from micromagpy import drivers
from micromagpy.config import BOLTZMANN, ELECTRON_CHARGE, HBAR
from micromagpy.noise import ThermalField
from micromagpy.torques import SOT


def test_constants_live_in_config():
    import micromagpy.noise as noise
    import micromagpy.torques as torques

    assert torques.HBAR is HBAR and torques.ELECTRON_CHARGE is ELECTRON_CHARGE
    assert noise.BOLTZMANN is BOLTZMANN


def test_as_driver_wraps_values_and_callables():
    c = drivers.as_driver([0, 0, 1e4])
    assert c.constant
    np.testing.assert_array_equal(c(1.0), [0, 0, 1e4])
    f = drivers.as_driver(lambda t: [t, 0, 0])
    assert not f.constant
    np.testing.assert_array_equal(f(2.0), [2.0, 0, 0])
    assert drivers.as_driver(c) is c


def test_step_pulse_sine():
    p = drivers.pulse(5.0, start=1.0, width=2.0)
    assert [float(p(t)) for t in (0.5, 1.0, 2.9, 3.0)] == [0.0, 5.0, 5.0, 0.0]
    s = drivers.sine(2.0, frequency=1.0, offset=1.0)
    assert np.isclose(float(s(0.25)), 3.0)


def test_drivers_in_torques_and_thermal_field():
    m = np.zeros((2, 1, 1, 3))
    m[..., 0] = 1.0
    sot = SOT(Hdl=drivers.pulse(1e3, 0.0, 1.0), p=(0, 1, 0))
    assert np.any(sot.rhs(m, 0.5, 0.0, 2.2e5) != 0)
    np.testing.assert_array_equal(sot.rhs(m, 2.0, 0.0, 2.2e5), 0.0)
    th = ThermalField((2, 1, 1), (5e-9,) * 3, 8e5, 0.1, drivers.step(300.0, 1.0), 1e-13)
    assert th.sigma(0.5) == 0.0 and th.sigma(1.5) > 0.0
