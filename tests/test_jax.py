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
