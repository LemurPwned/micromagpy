import numpy as np
import pytest

from micromagpy.config import GAMMA, MU0
from micromagpy.llg import ll_operator, pure_llg_torque
from micromagpy.simulation import simulate
from micromagpy.torques import ELECTRON_CHARGE, HBAR, SOT, STT, torque_rhs


def random_m(n, seed=0):
    m = np.random.default_rng(seed).normal(size=n + (3,))
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def test_field_like_sot_equals_external_field():
    # cmtj convention: a pure field-like SOT acts exactly like a field Hfl * p
    m, alpha, hfl, p = random_m((3, 3, 1)), 0.3, 2e4, np.array([0.0, 1.0, 0.0])
    tq = torque_rhs(m, 0.0, alpha, GAMMA, [SOT(Hfl=hfl, Hdl=0.0, p=p)])
    np.testing.assert_allclose(tq, ll_operator(alpha, GAMMA, m, hfl * p), rtol=1e-12, atol=1e-6)


def test_sot_matches_cmtj_formula():
    m, alpha, hfl, hdl = random_m((2, 2, 1), 1), 0.05, 1e3, 4e3
    p = np.array([0.0, 1.0, 0.0])
    got = torque_rhs(m, 0.0, alpha, GAMMA, [SOT(hfl, hdl, p)])
    mxp = np.cross(m, p)
    expected = -GAMMA / (1 + alpha**2) * ((hfl - alpha * hdl) * mxp + (hdl + alpha * hfl) * np.cross(m, mxp))
    np.testing.assert_allclose(got, expected, rtol=1e-12)


@pytest.mark.parametrize("alternative", [False, True])
def test_stt_matches_cmtj_formula(alternative):
    m, alpha = random_m((2, 2, 1), 2), 0.02
    p = np.array([0.0, 1.0, 1.0]) / np.sqrt(2)
    j, d, ms, P, lam, beta, kappa = -6e9, 1.4e-9, 1 / MU0, 0.8, 1.5, 0.3, 0.9
    stt = STT(j, p, d, ms, polarisation=P, spacer_lambda=lam, beta=beta, kappa=kappa, alternative=alternative)
    got = stt.rhs(m, 0.0, alpha, GAMMA)
    aJ = HBAR * j / (ELECTRON_CHARGE * (MU0 * ms) * d)  # cmtj uses Ms in tesla
    mp = m @ p
    eta = P / (1 + lam * mp) if alternative else P * lam**2 / (lam**2 + 1 + (lam**2 - 1) * mp)
    stt_term = GAMMA * aJ * eta
    fl = np.cross(m, p)
    dl = np.cross(m, fl)
    expected = (-stt_term * kappa)[..., None] * dl + (stt_term * beta)[..., None] * fl
    np.testing.assert_allclose(got, expected / (1 + alpha**2), rtol=1e-12)


def test_pure_llg_torque_consistent_with_simulate():
    n = (4, 4, 1)
    m = random_m(n, 3)
    h = np.array([0, 0, 1e5])
    sot = SOT(Hfl=3e3, Hdl=5e3, p=(1, 0, 0))
    a = pure_llg_torque(0.1, GAMMA, m, h, Hdl=5e3, Hfl=3e3, p=(1, 0, 0), dt=1e-13)
    b = simulate(m, (5e-9,) * 3, 8e5, 0.0, 0.1, 1e-13, 1, h_ext=h, demag=False, torques=[sot], solver="euler")["m"]
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_damping_like_sot_switches_macrospin():
    # PMA macrospin, damping-like SOT with p = y and bias field along x: deterministic switching
    n, dx, ms, K = (1, 1, 1), (10e-9, 10e-9, 1e-9), 8e5, 8e5
    m = np.zeros(n + (3,))
    m[..., 2] = 1.0
    Hk_eff = 2 * K / (MU0 * ms) - ms
    sot = SOT(Hdl=0.6 * Hk_eff, p=(0, 1, 0))
    common = dict(K=K, h_ext=(2e4, 0, 0), demag=True, solver="rk45", record_every=100)
    plus = simulate(m, dx, ms, 0.0, 0.1, 1e-12, 2000, torques=[sot], **common)["m_avg"][-1, 2]
    sot_neg = SOT(Hdl=-0.6 * Hk_eff, p=(0, 1, 0))
    minus = simulate(m, dx, ms, 0.0, 0.1, 1e-12, 2000, torques=[sot_neg], **common)["m_avg"][-1, 2]
    assert np.sign(plus) != np.sign(minus)


def test_torque_mask_and_time_driver():
    n = (4, 1, 1)
    m = random_m(n, 4)
    mask = np.array([True, False, True, False]).reshape(n)
    sot = SOT(Hdl=lambda t: 1e3 * t, p=(0, 1, 0), mask=mask)
    out = sot.rhs(m, 2.0, 0.0, GAMMA)
    np.testing.assert_array_equal(out[1], 0.0)
    assert np.all(np.abs(out[0]) > 0)
    np.testing.assert_allclose(sot.rhs(m, 4.0, 0.0, GAMMA)[0], 2 * out[0])
