"""The optional JAX backend must reproduce the numba path (float64)."""

import numpy as np
import pytest

pytest.importorskip("jax")

from micromagpy import jax_backend as jb  # noqa: E402

jb.enable_x64()

import jax.numpy as jnp  # noqa: E402

from micromagpy.config import GAMMA  # noqa: E402
from micromagpy.demag import compute_demag_tensor  # noqa: E402
from micromagpy.fields import h_eff  # noqa: E402
from micromagpy.llg import pure_llg, rk4, stochastic_heun  # noqa: E402

MESHES = [
    ((24, 10, 1), (5e-9, 5e-9, 3e-9)),
    ((10, 8, 4), (4e-9, 5e-9, 3e-9)),
    ((12, 1, 1), (5e-9, 5e-9, 5e-9)),
    ((1, 1, 1), (5e-9, 5e-9, 5e-9)),
]
A, MS, K, KAXIS, ALPHA, DT = 1.3e-11, 8e5, 3e5, (0.3, 0.1, 1.0), 0.05, 1e-13
H_EXT = np.array([2e3, 0.0, 1e4])


def _setup(n, dx, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=n + (3,))
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    f = compute_demag_tensor(n, dx)
    model = jb.JaxModel(n, dx, MS, A, ALPHA, K, KAXIS, f_n_demag=f)
    return m, f, model


@pytest.mark.parametrize("n,dx", MESHES)
def test_field_matches_numba(n, dx):
    m, f, model = _setup(n, dx)
    ref = h_eff(A, MS, K, KAXIS, m, None, f, n, dx)
    out = np.asarray(model.h_eff(jnp.asarray(m)))
    np.testing.assert_allclose(out, ref, rtol=0, atol=1e-12 * np.abs(ref).max())


@pytest.mark.parametrize("n,dx", MESHES[:2])
@pytest.mark.parametrize("solver", ["euler", "heun", "rk4"])
def test_scanned_run_matches_numba(n, dx, solver):
    m, f, model = _setup(n, dx, seed=1)
    steps = 100 if solver == "rk4" else 300

    def field(mm):
        return h_eff(A, MS, K, KAXIS, mm, None, f, n, dx)

    mr = m.copy()
    avgs = []
    for i in range(steps):
        if i % 50 == 0:
            avgs.append(mr.mean(axis=(0, 1, 2)))
        if solver == "euler":
            mr = pure_llg(ALPHA, GAMMA, mr, field(mr) + H_EXT, DT)
        elif solver == "heun":
            mr = stochastic_heun(ALPHA, GAMMA, mr, field, H_EXT, DT)
        else:
            mr = rk4(ALPHA, GAMMA, mr, lambda mm, t: field(mm) + H_EXT, i * DT, DT)
    out = model.run(m, DT, steps, solver=solver, h_ext=H_EXT, record_every=50)
    np.testing.assert_allclose(out["m"], mr, rtol=0, atol=1e-10)
    np.testing.assert_allclose(out["m_avg"], np.array(avgs), rtol=0, atol=1e-10)


def test_float32_close_to_float64():
    n, dx = MESHES[1]
    m, f, model = _setup(n, dx, seed=2)
    m32 = jb.JaxModel(n, dx, MS, A, ALPHA, K, KAXIS, f_n_demag=f, dtype=jnp.float32)
    h64 = np.asarray(model.h_eff(jnp.asarray(m)))
    h32 = np.asarray(m32.h_eff(jnp.asarray(m, jnp.float32)))
    assert h32.dtype == np.float32
    np.testing.assert_allclose(h32, h64, rtol=0, atol=1e-5 * np.abs(h64).max())
    r64 = model.run(m, DT, 200, solver="heun", h_ext=H_EXT, record_every=200)["m"]
    r32 = m32.run(m, DT, 200, solver="heun", h_ext=H_EXT, record_every=200)["m"]
    # random initial state: float32 round-off accumulates to ~1e-4 in 200 steps
    assert np.abs(r32 - r64).max() < 1e-3


def test_thermal_noise_statistics():
    """Brown's field drawn in the scan: macrospin <m_z> at equilibrium ~ Langevin."""
    n, dx = (1, 1, 1), (5e-9, 5e-9, 5e-9)
    T, Hz = 300.0, 1e5
    model = jb.JaxModel(n, dx, MS, 0.0, 0.5, demag=False, thermal_T=T)
    m0 = np.zeros(n + (3,))
    m0[..., 2] = 1.0
    out = model.run(m0, 1e-13, 40_000, solver="heun", h_ext=(0, 0, Hz), record_every=10, seed=3)
    from micromagpy.config import MU0
    from micromagpy.noise import BOLTZMANN

    xi = MU0 * MS * np.prod(dx) * Hz / (BOLTZMANN * T)
    langevin = 1 / np.tanh(xi) - 1 / xi
    assert out["m_avg"][500:, 2].mean() == pytest.approx(langevin, abs=0.02)


def test_ou_noise_generator_and_run():
    """1/f OU poles (taken from OUAlphaNoise) updated with jax.random keep their variance,
    and a scanned run with the coloured field is reproducible for a fixed seed."""
    import jax

    from micromagpy.noise import OUAlphaNoise

    gen = OUAlphaNoise(1.0, std=1.0, f_min=1e8, f_max=1e11, dt=1e-12, shape=())
    tau, sig = jnp.asarray(gen.tau), jnp.asarray(gen.sigma)
    a, b = jb.ou_coefficients(tau, sig, 1e-12)
    np.testing.assert_allclose(np.asarray(a), gen._a.ravel())
    np.testing.assert_allclose(np.asarray(b), gen._b.ravel())

    def body(x, key):
        x = jb.ou_update(x, a, b, key)
        return x, x.sum()

    keys = jax.random.split(jax.random.PRNGKey(0), 50_000)
    _, xs = jax.lax.scan(body, sig * jax.random.normal(jax.random.PRNGKey(1), sig.shape), keys)
    assert float(jnp.std(xs)) == pytest.approx(1.0, rel=0.15)

    n, dx = (4, 4, 1), (5e-9, 5e-9, 3e-9)
    ou = dict(tau=gen.tau, sigma=1e4 * gen.sigma, direction=(1, 0, 0))
    model = jb.JaxModel(n, dx, MS, A, 0.1, demag=True, ou=ou)
    m0 = np.zeros(n + (3,))
    m0[..., 2] = 1.0
    r1 = model.run(m0, 1e-13, 500, solver="heun", record_every=100, seed=7)
    r2 = model.run(m0, 1e-13, 500, solver="heun", record_every=100, seed=7)
    r3 = model.run(m0, 1e-13, 500, solver="heun", record_every=100, seed=8)
    np.testing.assert_array_equal(r1["m"], r2["m"])
    assert np.abs(r1["m"] - r3["m"]).max() > 1e-6
    assert np.abs(r1["m"][..., 0]).max() > 1e-4  # the x-directed field tilted m
