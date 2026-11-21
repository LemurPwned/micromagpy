import numpy as np
import pytest

from micromagpy.config import GAMMA, MU0
from micromagpy.llg import pure_llg, stochastic_heun
from micromagpy.noise import (
    BOLTZMANN,
    AnisotropyNoise,
    BufferedAlphaNoise,
    ColouredNoiseField,
    OneFNoise,
    OUAlphaNoise,
    ThermalField,
    VectorAlphaNoise,
    alpha_noise,
    fit_spectral_exponent,
    labels_from_masks,
    power_spectral_density,
    voronoi_grains,
)
from micromagpy.simulation import simulate

DT = 1e-12


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
def test_kasdin_spectral_exponent(alpha):
    x = alpha_noise(2**15, 1.0, alpha, size=(8,), seed=1)
    f, p = power_spectral_density(x, DT, axis=-1, segments=4)
    assert fit_spectral_exponent(f, p.mean(0), f_max=1e10) == pytest.approx(alpha, abs=0.1)


@pytest.mark.parametrize("alpha", [0.5, 1.0, 1.5])
def test_ou_spectral_exponent_and_rms(alpha):
    gen = OUAlphaNoise(alpha, 2.0, 1e6, 5e11, DT, shape=(8,), seed=2)
    x = np.array([gen.tick() for _ in range(2**15)])
    f, p = power_spectral_density(x, DT, segments=4)
    assert fit_spectral_exponent(f, p.mean(1), f_min=1e8, f_max=1e11) == pytest.approx(alpha, abs=0.1)
    # the band-limited theoretical PSD integrates to the requested variance
    ff = np.logspace(0, 16, 20000)
    assert np.sqrt(np.trapezoid(gen.theoretical_psd(ff), ff)) == pytest.approx(2.0, rel=0.01)


def test_ou_rejects_invalid_alpha():
    with pytest.raises(ValueError):
        OUAlphaNoise(2.5, 1.0, 1.0, 10.0, DT)


def test_onef_noise_is_pink_and_seeded():
    a = OneFNoise(20, 0.5, 1.0, shape=(8,), seed=3)
    b = OneFNoise(20, 0.5, 1.0, shape=(8,), seed=3)
    x = np.array([a.tick() for _ in range(2**15)])
    y = np.array([b.tick() for _ in range(2**15)])
    np.testing.assert_array_equal(x, y)
    f, p = power_spectral_density(x, DT, segments=4)
    assert fit_spectral_exponent(f, p.mean(1), f_max=1e10) == pytest.approx(1.0, abs=0.15)
    assert isinstance(OneFNoise(5, 0.5, 1.0, seed=0).tick(), float)


def test_buffered_noise_ticks_through_buffer():
    gen = BufferedAlphaNoise(64, 1.0, 1.0, scale=3.0, shape=(2,), seed=0)
    first = np.array([gen.tick() for _ in range(64)])
    np.testing.assert_allclose(first, 3.0 * gen.full_buffer())
    gen.tick()
    assert gen.refills == 2


def test_vector_alpha_noise():
    v = VectorAlphaNoise(128, 1.0, 0.1, scale=5.0, shape=(4,), seed=0)
    np.testing.assert_allclose(np.linalg.norm(v.tick_vector(), axis=-1), 5.0)
    vz = VectorAlphaNoise(128, 1.0, 0.1, scale=2.0, axis="z", seed=0)
    s = vz.tickVector()
    assert s[0] == 0 and s[1] == 0 and s[2] != 0


def test_thermal_field_amplitude_and_mask():
    n, dx = (10, 10, 1), (5e-9, 5e-9, 5e-9)
    mask = np.zeros(n, dtype=bool)
    mask[:5] = True
    th = ThermalField(n, dx, 8e5, 0.1, 300.0, 1e-13, mask=mask, seed=0)
    expected = np.sqrt(2 * 0.1 * BOLTZMANN * 300 / (GAMMA * MU0 * 8e5 * np.prod(dx) * 1e-13))
    assert th.sigma() == pytest.approx(expected)
    h = np.concatenate([th() for _ in range(50)])
    assert np.all(h[5:10] == 0)
    sel = np.concatenate([th()[:5] for _ in range(50)])
    assert sel.std() == pytest.approx(expected, rel=0.05)


def test_thermal_equilibrium_matches_langevin():
    # non-interacting cells in a field: <m_z> = L(xi) = coth(xi) - 1/xi
    n, dx, ms, T, xi = (12, 12, 1), (5e-9,) * 3, 8e5, 300.0, 3.0
    H = xi * BOLTZMANN * T / (MU0 * ms * np.prod(dx))
    m = np.zeros(n + (3,))
    m[..., 2] = 1
    th = ThermalField(n, dx, ms, 0.5, T, 1e-13, seed=1)
    r = simulate(m, dx, ms, 0.0, 0.5, 1e-13, 8000, h_ext=(0, 0, H), demag=False, noise=[th], record_every=10)
    mz = r["m_avg"][200:, 2].mean()
    assert mz == pytest.approx(1 / np.tanh(xi) - 1 / xi, abs=0.03)


def test_region_noise_structure():
    n = (12, 10, 1)
    labels = voronoi_grains(n, 6, seed=0)
    labels[0, 0, 0] = 0
    field = ColouredNoiseField(n, 100.0, 1.0, DT, regions=labels, seed=0)
    h = field.tick()
    np.testing.assert_array_equal(h[0, 0, 0], 0.0)
    for r in range(1, 7):
        sel = h[labels == r]
        np.testing.assert_allclose(sel, np.broadcast_to(sel[0], sel.shape))


def test_region_noise_volume_scaling_and_masks():
    n = (20, 20, 1)
    big = np.zeros(n, dtype=bool)
    big[:16, :16] = True
    small = np.zeros(n, dtype=bool)
    small[18:, 18:] = True  # 4 cells vs 256 cells
    labels = labels_from_masks([big, small], n)
    field = ColouredNoiseField(n, 1.0, 1.0, DT, regions=[big, small], volume_scaling=True, axis=(0, 0, 1), seed=0)
    np.testing.assert_array_equal(field.labels, labels)
    ratio = field._amp[2, 0] / field._amp[1, 0]
    assert ratio == pytest.approx(np.sqrt(256 / 4))
    h = field.tick()
    assert np.all(h[..., :2] == 0)


def test_correlated_cell_noise():
    n, dx, length = (64, 64, 1), (1e-9, 1e-9, 1e-9), 4e-9
    field = ColouredNoiseField(n, 1.0, 1.0, DT, correlation_length=length, dx=dx, axis=(1, 0, 0), seed=0)
    x = np.stack([field.tick()[..., 0, 0] for _ in range(20)])
    assert x.std() == pytest.approx(1.0, rel=0.15)
    c4 = np.mean(x * np.roll(x, 4, axis=1)) / np.mean(x * x)
    c0 = np.mean(x * np.roll(x, 16, axis=1)) / np.mean(x * x)
    assert c4 == pytest.approx(np.exp(-0.5), abs=0.1)
    assert abs(c0) < 0.1


def test_anisotropy_noise_field_direction():
    n = (4, 4, 1)
    rng = np.random.default_rng(0)
    m = rng.normal(size=n + (3,))
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    an = AnisotropyNoise(n, 8e5, 1e3, (0, 0, 1), 1.0, DT, seed=0)
    h = an.sample(m=m)
    np.testing.assert_array_equal(h[..., :2], 0.0)
    np.testing.assert_allclose(h[..., 2], 2 * an.delta_k() / (MU0 * 8e5) * m[..., 2])


def test_heun_converges_to_euler_without_noise():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(4, 4, 1, 3))
    m /= np.linalg.norm(m, axis=-1, keepdims=True)
    h = np.array([0, 0, 1e5])
    mh, me = m.copy(), m.copy()
    for _ in range(200):
        mh = stochastic_heun(0.1, GAMMA, mh, lambda mm: np.zeros_like(mm), h, 1e-14)
        me = pure_llg(0.1, GAMMA, me, h, 1e-14)
    np.testing.assert_allclose(mh, me, atol=1e-4)


def test_simulate_with_coloured_noise_runs_and_is_finite():
    n, dx = (8, 8, 1), (5e-9, 5e-9, 3e-9)
    m = np.zeros(n + (3,))
    m[..., 2] = 1
    noise = [
        ColouredNoiseField(n, 1e3, 1.0, 1e-13, regions=voronoi_grains(n, 4, seed=1), seed=0),
        AnisotropyNoise(n, 8e5, 1e3, (0, 0, 1), 1.0, 1e-13, seed=1),
        ThermalField(n, dx, 8e5, 0.1, 300.0, 1e-13, seed=2),
    ]
    r = simulate(m, dx, 8e5, 1.3e-11, 0.1, 1e-13, 50, K=5e5, noise=noise)
    assert np.all(np.isfinite(r["m"]))
    np.testing.assert_allclose(np.linalg.norm(r["m"], axis=-1), 1.0)


def test_rk4_is_stable_where_euler_blows_up():
    # 4 nm cells, low damping: Euler amplifies exchange spin waves, RK4 does not
    n, dx, ms = (8, 8, 1), (4e-9, 4e-9, 1e-9), 8.2e5
    m = np.zeros(n + (3,))
    m[..., 2] = 1.0
    tilt = np.deg2rad(1.0)
    h = 7e5 * np.array([np.sin(tilt), 0, np.cos(tilt)])
    kw = dict(K=489e3, h_ext=h, demag=True, record_every=100)
    rng = np.random.default_rng(0)
    m += 0.01 * rng.normal(size=m.shape)
    euler = simulate(m, dx, ms, 1.3e-11, 0.01, 1e-13, 3000, solver="euler", **kw)
    rk = simulate(m, dx, ms, 1.3e-11, 0.01, 1e-13, 3000, solver="rk4", **kw)
    assert rk["m_avg"][-1, 2] > 0.99
    assert euler["m_avg"][-1, 2] < 0.9


def test_rk4_rejects_thermal_noise():
    n, dx = (2, 2, 1), (5e-9,) * 3
    m = np.zeros(n + (3,))
    m[..., 2] = 1
    with pytest.raises(ValueError):
        simulate(m, dx, 8e5, 0, 0.1, 1e-13, 1, noise=[ThermalField(n, dx, 8e5, 0.1, 300, 1e-13)], solver="rk4")
