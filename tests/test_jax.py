"""The JAX backend must reproduce the numba path (fields, solvers, torques)."""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

from micromagpy import jax_backend as mj  # noqa: E402
from micromagpy.demag import compute_demag_tensor  # noqa: E402
from micromagpy.fields import h_eff  # noqa: E402
from micromagpy.simulation import simulate  # noqa: E402
from micromagpy.torques import SOT  # noqa: E402

DX = (4e-9, 4e-9, 2e-9)
MS, A, K = 8e5, 1.3e-11, 3e5


def _m(n, seed=0):
    m = np.random.default_rng(seed).normal(size=n + (3,))
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


@pytest.mark.parametrize("D", [0.0, 2e-3])
@pytest.mark.parametrize("n", [(12, 7, 1), (6, 5, 3), (1, 1, 1)])
def test_field_matches_numba(n, D):
    m = _m(n)
    if n[0] > 2:
        m[0, 0] = m[3, 2] = 0.0  # empty cells: DMI bonds to them are missing
    ref = h_eff(A, MS, K, (0.3, 0, 1), m, None, compute_demag_tensor(n, DX), n, DX, D=D)
    got = np.asarray(mj.make_field(n, DX, MS, A, K, (0.3, 0, 1), D=D)(m))
    np.testing.assert_allclose(got, ref, rtol=1e-9, atol=1e-9 * MS)


@pytest.mark.parametrize("solver", ["rk4", "heun"])
def test_trajectory_matches_simulate(solver):
    n, dt, steps, alpha = (8, 6, 1), 2e-14, 300, 0.05
    m0 = _m(n, 1)
    h = np.array([1e4, 0, 2e4])
    sot = SOT(Hfl=2e3, Hdl=5e3, p=(0, 1, 0))
    ref = simulate(m0, DX, MS, A, alpha, dt, steps, K=K, h_ext=h, solver=solver, record_every=10, torques=[sot], D=1e-3)
    field = mj.make_field(n, DX, MS, A, K, D=1e-3)
    out = mj.run(m0, field, alpha, dt, steps, h_ext=lambda t: h, extra=mj.sot(2e3, 5e3, (0, 1, 0)),
                 solver=solver, record_every=10)
    np.testing.assert_allclose(np.asarray(out["m_avg"]), ref["m_avg"], atol=1e-10)
    np.testing.assert_allclose(np.asarray(out["m"]), ref["m"], atol=1e-10)


def test_thermal_sigma_and_vmap():
    from micromagpy.noise import ThermalField

    n, dt = (4, 4, 1), 1e-13
    th = ThermalField(n, DX, MS, 0.01, 300.0, dt)
    assert mj.thermal_sigma(DX, MS, 0.01, 300.0, dt) == pytest.approx(th.sigma())
    field = mj.make_field(n, DX, MS, A, K)
    m0 = np.tile([0.0, 0.0, 1.0], n + (1,))
    run = jax.jit(jax.vmap(lambda key: mj.run(m0, field, 0.01, dt, 50, sigma=th.sigma(), key=key)["m"]))
    ms = np.asarray(run(jax.random.split(jax.random.PRNGKey(0), 3)))
    assert ms.shape == (3,) + n + (3,)
    np.testing.assert_allclose(np.linalg.norm(ms, axis=-1), 1.0, atol=1e-12)
    assert not np.allclose(ms[0], ms[1])  # independent noise per batch member


def test_float32_with_time_dependent_drive():
    jnp = jax.numpy
    n = (4, 4, 1)
    field = mj.make_field(n, DX, MS, A, K, dtype=jnp.float32)
    m0 = jnp.zeros(n + (3,), jnp.float32).at[..., 2].set(1.0)
    out = mj.run(m0, field, 0.05, 1e-13, 20, h_ext=lambda t: jnp.array([1e4, 0, 0]) * jnp.sin(1e10 * t),
                 extra=mj.sot(lambda t: 1e3 * t / 1e-12, 2e3, (0, 1, 0)), solver="rk4")
    assert out["m"].dtype == jnp.float32


def test_stt_matches_numba():
    from micromagpy.torques import STT

    n, alpha = (5, 4, 1), 0.02
    m = _m(n, 3)
    for kw in (dict(polarisation=0.6, spacer_lambda=1.5, beta=0.2), dict(polarisation=0.4, alternative=True, kappa=0.7)):
        ref = STT(3e10, (0, 1, 1), 2e-9, MS, **kw).rhs(m, 0.0, alpha, 2.21e5)
        got = mj.stt(3e10, (0, 1, 1), 2e-9, MS, **kw)(jax.numpy.asarray(m), 0.0, alpha, 2.21e5 / (1 + alpha**2))
        np.testing.assert_allclose(np.asarray(got), ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def test_rk45_matches_simulate():
    from micromagpy.torques import STT

    n, dt, steps, alpha = (8, 6, 1), 1e-12, 60, 0.02
    m0 = _m(n, 2)
    h = np.array([1e4, 0, 2e4])
    torques = [SOT(Hfl=2e3, Hdl=5e3, p=(0, 1, 0)), STT(2e10, (0, 0, 1), DX[2], MS, polarisation=0.5)]
    ref = simulate(m0, DX, MS, A, alpha, dt, steps, K=K, h_ext=h, solver="rk45", tol=1e-6, record_every=5,
                   torques=torques, D=1e-3)
    sot, stt = mj.sot(2e3, 5e3, (0, 1, 0)), mj.stt(2e10, (0, 0, 1), DX[2], MS, polarisation=0.5)
    out = mj.run(m0, mj.make_field(n, DX, MS, A, K, D=1e-3), alpha, dt, steps, h_ext=lambda t: h,
                 extra=lambda m, t, a, p: sot(m, t, a, p) + stt(m, t, a, p), solver="rk45", tol=1e-6, record_every=5)
    assert int(out["rk45_steps"]) == ref["rk45_steps"]
    np.testing.assert_allclose(np.asarray(out["m_avg"]), ref["m_avg"], atol=1e-9)
    np.testing.assert_allclose(np.asarray(out["m"]), ref["m"], atol=1e-9)


def test_relax_matches_numba():
    from micromagpy.energy import relax

    n = (10, 8, 1)
    m0 = _m(n, 4)
    m0[0, 0] = 0.0  # an empty cell
    h = (2e4, 0, 1e4)
    ref = relax(m0, DX, MS, A=A, K=K, h_ext=h, torque_tol=1e-4, tol=1e-6, D=1e-3)
    got = mj.relax(m0, mj.make_field(n, DX, MS, A, K, D=1e-3), MS, h_ext=h, torque_tol=1e-4, tol=1e-6)
    assert bool(got["converged"]) and ref["converged"]
    assert int(got["steps"]) == ref["steps"]
    np.testing.assert_allclose(np.asarray(got["m"]), ref["m"], atol=1e-9)


def test_run_noise_source_is_a_held_field():
    n, dt = (6, 5, 1), 1e-13
    m0 = _m(n, 5)
    H = np.random.default_rng(1).normal(size=n + (3,)) * 3e4
    field = mj.make_field(n, DX, MS, A, K)
    const = (jax.numpy.zeros(()), lambda st, key: (st, jax.numpy.asarray(H)))
    for solver in ("heun", "rk4"):
        a = mj.run(m0, field, 0.02, dt, 50, h_ext=lambda t: H, solver=solver)
        b = mj.run(m0, field, 0.02, dt, 50, noise=[const], solver=solver)
        np.testing.assert_allclose(np.asarray(b["m"]), np.asarray(a["m"]), atol=1e-14)


def test_coloured_noise_matches_numpy_source():
    from micromagpy.noise import ColouredNoiseField, voronoi_grains

    n, dt, amp = (12, 10, 1), 1e-12, 2e3
    labels = voronoi_grains(n, 5, seed=0)
    labels[0, 0] = 0  # one cell without noise
    src = ColouredNoiseField(n, amp, 1.0, dt, regions=labels, generator_kwargs=dict(f_min=1e-2 / dt), seed=3)
    state, step = mj.coloured_noise(src)
    keys = jax.random.split(jax.random.PRNGKey(0), 20000)
    h = np.asarray(jax.jit(lambda s: jax.lax.scan(step, s, keys)[1])(state))
    assert np.all(h[:, 0, 0] == 0)
    for r in range(1, 6):  # one vector per grain, rms = amplitude per component
        cells = h[:, labels == r]
        np.testing.assert_array_equal(cells, np.broadcast_to(cells[:, :1], cells.shape))
        np.testing.assert_allclose(cells[:, 0].std(0), amp, rtol=0.1)
