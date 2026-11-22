"""Interfacial DMI: energy consistency, bulk field and the analytic edge canting."""

import numpy as np
from scipy.optimize import minimize

from micromagpy.config import MU0
from micromagpy.energy import energies
from micromagpy.fields import compute_anisotropy_field, compute_hdmi, compute_hex

MS, A = 1e6, 1.5e-11


def test_field_is_energy_gradient():
    """The bond discretisation is a symmetric linear operator: a . H(b) = b . H(a), also with empty cells."""
    rng = np.random.default_rng(0)
    n, dx = (7, 5, 2), (1e-9, 1.5e-9, 1e-9)
    a, b = rng.normal(size=n + (3,)), rng.normal(size=n + (3,))
    a[2, 2] = b[2, 2] = 0.0
    lhs, rhs = np.sum(a * compute_hdmi(3e-3, MS, b, dx)), np.sum(b * compute_hdmi(3e-3, MS, a, dx))
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12)


def test_bulk_field_and_single_cell():
    """Inside the magnet: central difference of H = 2D/(mu0 Ms) (dmz/dx, dmz/dy, -dmx/dx - dmy/dy)."""
    D, d, k = 2e-3, 1e-9, 2 * np.pi / 20e-9
    x = np.arange(40) * d
    m = np.zeros((40, 3, 1, 3))
    m[..., 0], m[..., 2] = np.sin(k * x)[:, None, None], np.cos(k * x)[:, None, None]
    h = compute_hdmi(D, MS, m, (d, d, d))[1:-1, 1]
    c = 2 * D / (MU0 * MS) * np.sin(k * d) / d  # exact discrete derivative factor
    np.testing.assert_allclose(h[:, 0, 0], -c * np.sin(k * x[1:-1]), atol=1e-6 * c)
    np.testing.assert_allclose(h[:, 0, 2], -c * np.cos(k * x[1:-1]), atol=1e-6 * c)
    np.testing.assert_allclose(h[:, 0, 1], 0.0, atol=1e-6 * c)
    assert not np.any(compute_hdmi(D, MS, np.array([[[[0.3, 0.4, np.sqrt(0.75)]]]]), (d, d, d)))


def _edge_angle(D, K, d, L=100e-9):
    """Relaxed edge tilt (deg) of a PMA chain along x, by L-BFGS on the angles (m in the xz plane)."""
    N, dx = round(L / d), (d, d, d)
    n = (N, 1, 1)

    def fun(th):
        m = np.stack([np.sin(th), 0 * th, np.cos(th)], -1).reshape(n + (3,))
        e = energies(m, dx, MS, A=A, K=K, demag=False, D=D)["total"]
        h = compute_hex(A, MS, m, dx, n) + compute_hdmi(D, MS, m, dx) + compute_anisotropy_field(MS, K, m, (0, 0, 1))
        g = -MU0 * MS * d**3 * np.sum(h.reshape(N, 3) * np.stack([np.cos(th), 0 * th, -np.sin(th)], -1), -1)
        return e / (A * d), g / (A * d)

    r = minimize(fun, np.full(N, 0.01), jac=True, method="L-BFGS-B", options=dict(maxiter=50000, gtol=1e-12, ftol=1e-16))
    return np.degrees(r.x[0]), np.degrees(r.x[-1])


def test_edge_canting_matches_rohart_thiaville():
    """sin(theta_edge) = D / (2 sqrt(A K)) (Rohart & Thiaville, PRB 88, 184422); first order in the cell size."""
    D, K = 2e-3, 8e5
    exact = np.degrees(np.arcsin(D / (2 * np.sqrt(A * K))))
    (a1, b1), (a2, b2) = _edge_angle(D, K, 0.5e-9), _edge_angle(D, K, 0.25e-9)
    np.testing.assert_allclose([a1, a2], [-b1, -b2], atol=1e-4)  # the two edges tilt in opposite directions
    assert abs(a2 - exact) < abs(a1 - exact) < 0.1 * exact
    assert abs((2 * a2 - a1) - exact) < 0.05  # Richardson extrapolation
