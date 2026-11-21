import numpy as np
import pytest

from micromagpy.config import GAMMA, MU0
from micromagpy.demag import (
    compute_demag_tensor,
    compute_demag_tensor_real,
    compute_trivial_tensor,
)
from micromagpy.fields import (
    compute_anisotropy_field,
    compute_hdemag,
    compute_hex,
    h_eff,
)
from micromagpy.llg import ll_operator, pure_llg, pure_llg_torque

from . import _reference as ref

MESHES = [
    ((6, 5, 3), (2e-9, 3e-9, 1.5e-9)),
    ((8, 4, 1), (5e-9, 5e-9, 3e-9)),
    ((1, 1, 1), (5e-9, 5e-9, 5e-9)),
]


def random_m(n, seed=0):
    rng = np.random.default_rng(seed)
    m = rng.normal(size=n + (3,))
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


@pytest.mark.parametrize("n,dx", MESHES)
def test_demag_tensor_matches_reference(n, dx):
    # the reference is evaluated in units of the cell size (same as the new code),
    # it is otherwise inaccurate at nm scale because of the absolute eps in f/g
    scale = max(dx)
    expected = ref.demag_tensor_real(n, tuple(d / scale for d in dx))
    got = compute_demag_tensor_real(n, dx, shape=tuple(2 * k - 1 for k in n))
    np.testing.assert_allclose(got, expected, atol=1e-10 * np.abs(expected).max())


@pytest.mark.parametrize("scale", [1.0, 1e-6, 1e-9, 1e-10])
def test_demag_tensor_is_scale_invariant(scale):
    # self-demag factors of a single cell sum to 1 (N has a minus sign here)
    n_real = compute_demag_tensor_real((1, 1, 1), (2 * scale, 3 * scale, 5 * scale))
    assert np.isclose(
        n_real[0, 0, 0, 0] + n_real[0, 0, 0, 3] + n_real[0, 0, 0, 5], -1.0, atol=1e-12
    )
    cube = compute_demag_tensor_real((1, 1, 1), (scale,) * 3)[0, 0, 0]
    np.testing.assert_allclose(cube[[0, 3, 5]], -1 / 3, atol=1e-12)
    np.testing.assert_allclose(cube[[1, 2, 4]], 0.0, atol=1e-12)


@pytest.mark.parametrize("n,dx", MESHES)
def test_hdemag_matches_reference(n, dx):
    m = random_m(n)
    expected = ref.hdemag(
        m, ref.demag_tensor_real(n, tuple(d / max(dx) for d in dx)), n
    )
    got = compute_hdemag(m, compute_demag_tensor(n, dx), n)
    np.testing.assert_allclose(got, expected, atol=1e-10)


def test_hdemag_accepts_padded_input():
    n, dx = MESHES[0]
    m = random_m(n)
    m_pad = np.zeros([2 * k - 1 for k in n] + [3])
    m_pad[: n[0], : n[1], : n[2]] = m
    f = compute_demag_tensor(n, dx)
    np.testing.assert_allclose(compute_hdemag(m_pad, f, n), compute_hdemag(m, f, n))


def test_trivial_tensor_gives_zero_field():
    n = (8, 4, 1)
    np.testing.assert_array_equal(
        compute_hdemag(random_m(n), compute_trivial_tensor(n), n), 0.0
    )


@pytest.mark.parametrize("n,dx", MESHES)
def test_exchange_matches_reference(n, dx):
    m = random_m(n)
    np.testing.assert_allclose(
        compute_hex(1.3e-11, 8e5, m, dx, n),
        ref.hex_field(1.3e-11, 8e5, m, dx, n),
        rtol=1e-12,
        atol=1e-6,
    )


def test_anisotropy_field_general_axis():
    m = random_m((3, 3, 1))
    u = np.array([1.0, 1.0, 0.0]) / np.sqrt(2)
    h = compute_anisotropy_field(8e5, 1e5, m, (1, 1, 0))
    expected = 2 * 1e5 / (MU0 * 8e5) * (m @ u)[..., None] * u
    np.testing.assert_allclose(h, expected)


def test_heff_composition():
    n, dx = MESHES[0]
    m = random_m(n)
    f = compute_demag_tensor(n, dx)
    expected = (
        8e5 * compute_hdemag(m, f, n)
        + compute_hex(1e-11, 8e5, m, dx, n)
        + compute_anisotropy_field(8e5, 2e5, m, (0, 0, 1))
    )
    np.testing.assert_allclose(
        h_eff(1e-11, 8e5, 2e5, (0, 0, 1), m, None, f, n, dx),
        expected,
        rtol=1e-12,
        atol=1e-6,
    )


def test_llg_matches_reference():
    n = (5, 4, 2)
    m = random_m(n)
    h = random_m(n, 1) * 1e5
    np.testing.assert_allclose(
        pure_llg(0.1, GAMMA, m, h, 1e-13),
        ref.pure_llg(0.1, GAMMA, m, h, 1e-13),
        atol=1e-13,
    )


def test_ll_operator_is_perpendicular_to_m():
    m = random_m((5, 4, 2))
    dm = ll_operator(0.3, GAMMA, m, random_m((5, 4, 2), 2) * 1e4)
    np.testing.assert_allclose(np.sum(dm * m, axis=-1), 0.0, atol=1e-6)


def test_llg_keeps_empty_cells_empty():
    m = random_m((4, 4, 1))
    m[0, 0] = 0.0
    out = pure_llg(0.1, GAMMA, m, np.ones_like(m) * 1e4, 1e-13)
    assert np.all(np.isfinite(out))
    np.testing.assert_array_equal(out[0, 0], 0.0)


def test_torque_reduces_to_llg():
    m = random_m((4, 4, 1))
    h = random_m((4, 4, 1), 3) * 1e4
    np.testing.assert_allclose(
        pure_llg_torque(0.1, GAMMA, m, h, 0.0, 0.0, (1, 0, 0), 1e-13),
        pure_llg(0.1, GAMMA, m, h, 1e-13),
    )


def test_per_cell_anisotropy():
    n = (4, 3, 1)
    m = random_m(n, 5)
    K = np.linspace(1e4, 5e5, np.prod(n)).reshape(n)
    axes = random_m(n, 6)
    h = compute_anisotropy_field(8e5, K, m, axes)
    expected = (2 * K / (MU0 * 8e5) * np.sum(m * axes, axis=-1))[..., None] * axes
    np.testing.assert_allclose(h, expected, rtol=1e-12)
    f = compute_trivial_tensor(n)
    np.testing.assert_allclose(h_eff(0.0, 8e5, K, axes, m, None, f, n, (5e-9,) * 3), expected, rtol=1e-12)
