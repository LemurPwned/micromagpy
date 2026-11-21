import numpy as np
import pytest

from micromagpy.config import GAMMA, MU0
from micromagpy.energy import energies, relax
from micromagpy.solvers import RK45State, integrate_rk45, llg_rhs_factory
from micromagpy.simulation import simulate


def test_rk45_macrospin_precession_matches_analytic():
    # undamped precession about z: dm/dt = -gamma m x H rotates m_x + i m_y at +gamma H
    H, alpha = 1e5, 0.0
    m = np.zeros((1, 1, 1, 3))
    m[..., 0] = 1.0
    rhs = llg_rhs_factory(alpha, GAMMA, lambda mm, t: np.array([0, 0, H]))
    T = 1e-10
    out = integrate_rk45(rhs, m, 0.0, T, RK45State(dt=1e-13), tol=1e-9)
    phi = GAMMA * H * T
    np.testing.assert_allclose(out[0, 0, 0], [np.cos(phi), np.sin(phi), 0.0], atol=1e-6)


def test_rk45_simulate_reports_steps_and_is_stable():
    n, dx = (8, 8, 1), (4e-9, 4e-9, 1e-9)
    m = np.zeros(n + (3,))
    m[..., 2] = 1.0
    m += 0.01 * np.random.default_rng(0).normal(size=m.shape)
    tilt = np.deg2rad(1.0)
    h = 7e5 * np.array([np.sin(tilt), 0, np.cos(tilt)])
    r = simulate(m, dx, 8.2e5, 1.3e-11, 0.01, 1e-12, 200, K=489e3, h_ext=h, solver="rk45", tol=1e-5)
    assert r["m_avg"][-1, 2] > 0.99
    assert r["rk45_steps"] > 0


def test_energy_terms_single_domain():
    n, dx, ms, K = (2, 2, 1), (5e-9,) * 3, 8e5, 1e5
    m = np.zeros(n + (3,))
    m[..., 2] = 1.0
    V = np.prod(dx) * 4
    e = energies(m, dx, ms, A=1e-11, K=K, kaxis=(0, 0, 1), h_ext=(0, 0, 1e4), demag=False)
    assert e["exchange"] == pytest.approx(0.0, abs=1e-30)
    assert e["anisotropy"] == pytest.approx(-K * V)
    assert e["zeeman"] == pytest.approx(-MU0 * ms * 1e4 * V)


def test_relax_decreases_energy_and_converges():
    n, dx = (40, 10, 1), (5e-9, 5e-9, 3e-9)
    m = np.zeros(n + (3,))
    m[1:-1, :, :, 0] = 1.0
    m[(-1, 0), :, :, 1] = 1.0
    r = relax(m, dx, 8e5, A=1.3e-11, energy_every=5, torque_tol=1e-3)
    assert r["converged"]
    hist = np.array(r["energy_history"])
    assert np.all(np.diff(hist) <= 1e-6 * abs(hist[0]))
    assert r["m"].mean(axis=(0, 1, 2))[0] > 0.9
