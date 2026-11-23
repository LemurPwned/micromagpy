import jax
import jax.numpy as jnp
import numpy as np

from micromagpy import jax_backend as mj
from micromagpy.config import MU0
from micromagpy.demag import compute_demag_tensor_real, compute_layer_pair_tensors_real
from micromagpy.fft import padded_shape
from micromagpy.stack import Bar, Ellipse, Layer, Spacer, Stack

CELL = (4e-9, 3e-9)


def _random_m(stack, seed=0):
    m = np.random.default_rng(seed).normal(size=stack.masks.shape + (3,))
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    return jnp.asarray(m * stack.masks[..., None])


def test_pair_tensors_match_mesh_and_reciprocity():
    n, dx = (6, 5, 3), (3e-9, 4e-9, 2e-9)
    P = padded_shape(n)
    ref = compute_demag_tensor_real(n, dx, shape=P)
    pair = compute_layer_pair_tensors_real(n[:2], dx, [2e-9, 2e-9], [0.0, 4e-9], shape=P[:2])
    np.testing.assert_allclose(pair[0, 0], ref[:, :, 0], atol=1e-14)
    np.testing.assert_allclose(pair[0, 1], ref[:, :, -2 % P[2]], atol=1e-14)
    np.testing.assert_allclose(pair[1, 0], ref[:, :, 2], atol=1e-14)
    # unequal thickness: V_i N_ij(r) = V_j N_ji(-r)
    t = np.array([1e-9, 3e-9])
    p = compute_layer_pair_tensors_real((4, 3), (3e-9, 3e-9), t, [0.0, 3.5e-9])
    flip = lambda a: np.roll(a[::-1, ::-1], 1, (0, 1))  # noqa: E731  (r -> -r on the wrap grid)
    np.testing.assert_allclose(t[0] * p[0, 1], t[1] * flip(p[1, 0]), atol=1e-13)


def test_single_layer_stack_matches_make_field():
    st = Stack([Layer(Ellipse(40e-9, 24e-9), 2e-9, ms=8e5, A=1.3e-11, K=3e5, kaxis=(1, 0, 1))], CELL)
    m = _random_m(st)
    ref = mj.make_field(st.n + (1,), (*CELL, 2e-9), 8e5, 1.3e-11, K=3e5, kaxis=(1, 0, 1))(m[0][:, :, None])[:, :, 0]
    np.testing.assert_allclose(np.asarray(st.field()(m)[0]), np.asarray(ref) * st.masks[0][..., None], atol=1e-8)


def test_two_layers_match_mesh_with_empty_middle():
    t = 2e-9
    a, b = Layer(Bar(24e-9, 18e-9), t, ms=8e5, A=1.3e-11, K=2e5), Layer(Bar(24e-9, 18e-9), t, ms=8e5, A=1.3e-11, K=2e5)
    st = Stack([a, Spacer(t), b], CELL)
    m = _random_m(st, 1)
    mesh = jnp.stack([m[0], jnp.zeros_like(m[0]), m[1]], 2)  # (nx, ny, 3 cells, 3)
    # exchange along z couples nothing (middle is empty, Neumann edge on the others): compare without it
    ref = mj.make_field(st.n + (3,), (*CELL, t), 8e5, 1.3e-11, K=2e5)(mesh)
    lap_z = (jnp.stack([mesh[:, :, 0], mesh[:, :, 0], mesh[:, :, 1]], 2) + jnp.stack([mesh[:, :, 1], mesh[:, :, 2], mesh[:, :, 2]], 2) - 2 * mesh)
    ref = ref - 2 * 1.3e-11 / (MU0 * 8e5) * lap_z / t**2
    got = st.field()(m)
    np.testing.assert_allclose(np.asarray(got[0]), np.asarray(ref[:, :, 0]), atol=1e-8)
    np.testing.assert_allclose(np.asarray(got[1]), np.asarray(ref[:, :, 2]), atol=1e-8)


def test_iec_field_is_energy_gradient():
    st = Stack([Layer(Ellipse(32e-9, 32e-9), 1e-9, ms=8e5), Spacer(0.8e-9, J1=-1e-3, J2=-3e-4),
                Layer(Ellipse(24e-9, 24e-9), 2e-9, ms=1.1e6)], CELL)
    m = _random_m(st, 2)
    h_iec = st.field()(m) - st.field(J1=0.0, J2=0.0)(m)
    grad = jax.grad(lambda mm: st.energy_iec(mm).sum())(m)  # dE/dm
    V = (CELL[0] * CELL[1] * st.thickness * st.ms * MU0).reshape(-1, 1, 1, 1)
    np.testing.assert_allclose(np.asarray(h_iec), np.asarray(-grad / V) * st.masks[..., None], rtol=1e-10, atol=1e-6)


def test_iec_sign_and_j1_sweeps_under_vmap():
    pillar = Ellipse(32e-9, 32e-9)
    st = Stack([Layer(pillar, 1e-9, ms=8e5, A=1.3e-11, K=1.2e6), Spacer(0.8e-9, J1=-2e-3),
                Layer(pillar, 1e-9, ms=8e5, A=1.3e-11, K=6e5)], (4e-9, 4e-9))
    m0 = st.uniform([(0, 0, 1), (1, 0, 0.01)])  # stiff bottom up, top in-plane
    mz = jax.jit(jax.vmap(lambda j: st.layer_average(st.relax(m0, J1=j, tol=1e-6)["m"])[:, 2]))(jnp.array([-2e-3, 2e-3]))
    np.testing.assert_allclose(np.asarray(mz), [[1, -1], [1, 1]], atol=1e-3)  # J1 < 0 antiparallel, J1 > 0 parallel
