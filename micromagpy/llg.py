"""Landau-Lifshitz-Gilbert time stepping.

The right-hand side, the explicit update and the renormalisation of ``m`` are
fused into single numba kernels; this removes the four ``np.cross`` calls and
the temporaries of the pure NumPy version.
"""

from math import sqrt

import numpy as np
from numba import njit, prange

from .config import DTYPE, GAMMA
from .fields import h_eff


@njit(parallel=True, cache=True)
def _ll_rhs(m, h, pref, alpha, out):
    """out = -pref (m x h) - alpha pref m x (m x h)."""
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                mx, my, mz = m[i, j, k, 0], m[i, j, k, 1], m[i, j, k, 2]
                hx, hy, hz = h[i, j, k, 0], h[i, j, k, 1], h[i, j, k, 2]
                cx = my * hz - mz * hy
                cy = mz * hx - mx * hz
                cz = mx * hy - my * hx
                ddx = my * cz - mz * cy
                ddy = mz * cx - mx * cz
                ddz = mx * cy - my * cx
                out[i, j, k, 0] = -pref * (cx + alpha * ddx)
                out[i, j, k, 1] = -pref * (cy + alpha * ddy)
                out[i, j, k, 2] = -pref * (cz + alpha * ddz)


@njit(parallel=True, cache=True)
def _llg_torque_euler(m, h, pref, alpha, hfl, hdl, p, dt, out):
    """Explicit Euler step of LLG + torque ``afl m x p + adl m x (m x p)``, then renorm.

    ``hfl`` / ``hdl`` here are the combined coefficients ``a_fl`` / ``a_dl``
    (see :mod:`micromagpy.torques`). Cells with ``|m| == 0`` are left untouched.
    """
    nx, ny, nz, _ = m.shape
    px, py, pz = p[0], p[1], p[2]
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                mx, my, mz = m[i, j, k, 0], m[i, j, k, 1], m[i, j, k, 2]
                hx, hy, hz = h[i, j, k, 0], h[i, j, k, 1], h[i, j, k, 2]
                cx = my * hz - mz * hy
                cy = mz * hx - mx * hz
                cz = mx * hy - my * hx
                ddx = my * cz - mz * cy
                ddy = mz * cx - mx * cz
                ddz = mx * cy - my * cx
                # m x p and m x (m x p)
                qx = my * pz - mz * py
                qy = mz * px - mx * pz
                qz = mx * py - my * px
                rx = my * qz - mz * qy
                ry = mz * qx - mx * qz
                rz = mx * qy - my * qx
                nmx = mx + dt * (
                    -pref * (cx + alpha * ddx) - pref * (hfl * qx + hdl * rx)
                )
                nmy = my + dt * (
                    -pref * (cy + alpha * ddy) - pref * (hfl * qy + hdl * ry)
                )
                nmz = mz + dt * (
                    -pref * (cz + alpha * ddz) - pref * (hfl * qz + hdl * rz)
                )
                norm = sqrt(nmx * nmx + nmy * nmy + nmz * nmz)
                if norm > 0.0:
                    out[i, j, k, 0] = nmx / norm
                    out[i, j, k, 1] = nmy / norm
                    out[i, j, k, 2] = nmz / norm
                else:
                    out[i, j, k, 0] = nmx
                    out[i, j, k, 1] = nmy
                    out[i, j, k, 2] = nmz


@njit(parallel=True, cache=True)
def _normalize(m):
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                norm = sqrt(
                    m[i, j, k, 0] ** 2 + m[i, j, k, 1] ** 2 + m[i, j, k, 2] ** 2
                )
                if norm > 0.0:
                    m[i, j, k, 0] /= norm
                    m[i, j, k, 1] /= norm
                    m[i, j, k, 2] /= norm


_ZERO3 = np.zeros(3, dtype=DTYPE)


def _as_field(h, shape):
    h = np.asarray(h, dtype=DTYPE)
    if h.shape != shape:
        h = np.broadcast_to(h, shape)
    return np.ascontiguousarray(h)


def normalize(m: np.ndarray) -> np.ndarray:
    """Normalise ``m`` in place (zero cells are left as zero) and return it."""
    _normalize(m)
    return m


def ll_operator(alpha: float, gamma: float, m: np.ndarray, h) -> np.ndarray:
    """Landau-Lifshitz right-hand side dm/dt for field ``h`` (A/m)."""
    m = np.ascontiguousarray(m, dtype=DTYPE)
    out = np.empty_like(m)
    _ll_rhs(m, _as_field(h, m.shape), gamma / (1 + alpha**2), alpha, out)
    return out


def pure_llg(alpha: float, gamma: float, m: np.ndarray, h, dt: float) -> np.ndarray:
    """One explicit Euler step of the LLG equation (returns a new array)."""
    m = np.ascontiguousarray(m, dtype=DTYPE)
    out = np.empty_like(m)
    _llg_torque_euler(
        m,
        _as_field(h, m.shape),
        gamma / (1 + alpha**2),
        alpha,
        0.0,
        0.0,
        _ZERO3,
        dt,
        out,
    )
    return out


def pure_llg_torque(
    alpha: float, gamma: float, m: np.ndarray, h, Hdl: float, Hfl: float, p, dt: float
) -> np.ndarray:
    """Explicit Euler step of LLG with field-like and damping-like SOT (A/m).

    Same convention as cmtj (and :class:`micromagpy.torques.SOT`):
    ``-gamma' [(Hfl - alpha Hdl) m x p + (Hdl + alpha Hfl) m x (m x p)]``.
    """
    m = np.ascontiguousarray(m, dtype=DTYPE)
    out = np.empty_like(m)
    p = np.asarray(p, dtype=DTYPE).reshape(3)
    afl, adl = Hfl - alpha * Hdl, Hdl + alpha * Hfl
    _llg_torque_euler(
        m, _as_field(h, m.shape), gamma / (1 + alpha**2), alpha, afl, adl, p, dt, out
    )
    return out


def euler_heun(
    alpha: float,
    gamma: float,
    m: np.ndarray,
    h: np.ndarray,
    h_noise: np.ndarray,
    dt: float,
):
    """Euler-Heun step: Euler for the drift, Heun for the stochastic part.

    ``h_noise`` is the stochastic field already scaled such that its
    contribution over a step is ``g(m, h_noise) * sqrt(dt)`` (cmtj convention).
    """
    fn = ll_operator(alpha=alpha, gamma=gamma, m=m, h=h)
    gn = ll_operator(alpha=alpha, gamma=gamma, m=m, h=h_noise)
    m_next = m + gn * sqrt(dt)
    gnapprox = ll_operator(alpha=alpha, gamma=gamma, m=m_next, h=h_noise)
    m = m + fn * dt + 0.5 * (gn + gnapprox) * sqrt(dt)
    return normalize(m)


def stochastic_heun(
    alpha: float,
    gamma: float,
    m: np.ndarray,
    field_fn,
    h_fixed,
    dt: float,
    extra_rhs=None,
) -> np.ndarray:
    """Heun (predictor-corrector) step, the standard Stratonovich scheme for sLLG.

    :param field_fn: ``field_fn(m)`` -> m-dependent field (A/m), evaluated at
        the start of the step and at the predictor.
    :param h_fixed: field held constant during the step (A/m): external
        field, thermal field (already ~ 1/sqrt(dt)), coloured noise, ...
        May be ``None`` or a scalar ``0``.
    :param extra_rhs: optional ``extra_rhs(m, out)`` adding further dm/dt
        terms (e.g. SOT / STT torques) in place.
    """
    m = np.ascontiguousarray(m, dtype=DTYPE)
    pref = gamma / (1 + alpha**2)

    def total(mm):
        h = field_fn(mm)
        if h_fixed is not None and not (np.isscalar(h_fixed) and h_fixed == 0):
            h = h + h_fixed
        return _as_field(h, m.shape)

    k1 = np.empty_like(m)
    _ll_rhs(m, total(m), pref, alpha, k1)
    if extra_rhs is not None:
        extra_rhs(m, k1)
    m_pred = normalize(m + dt * k1)
    k2 = np.empty_like(m)
    _ll_rhs(m_pred, total(m_pred), pref, alpha, k2)
    if extra_rhs is not None:
        extra_rhs(m_pred, k2)
    return normalize(m + 0.5 * dt * (k1 + k2))


def rk4(
    alpha: float, gamma: float, m: np.ndarray, field_fn, t: float, dt: float, extra_rhs=None
) -> np.ndarray:
    """Classical Runge-Kutta step for the deterministic LLG equation.

    :param field_fn: ``field_fn(m, t)`` -> total field (A/m).
    :param extra_rhs: optional ``extra_rhs(m, t, out)`` adding dm/dt terms in place.

    Explicit Euler (and, more weakly, Heun) amplify undamped precession by
    ~(omega dt)^2 / 2 per step, which for low damping and small cells (high
    exchange spin-wave frequencies) exceeds the damping and blows up. RK4 is
    stable up to omega dt ~ 2.8. Not suitable for white (thermal) noise.
    """
    m = np.ascontiguousarray(m, dtype=DTYPE)
    pref = gamma / (1 + alpha**2)

    def rhs(mm, tt):
        out = np.empty_like(m)
        _ll_rhs(mm, _as_field(field_fn(mm, tt), m.shape), pref, alpha, out)
        if extra_rhs is not None:
            extra_rhs(mm, tt, out)
        return out

    k1 = rhs(m, t)
    k2 = rhs(m + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = rhs(m + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = rhs(m + dt * k3, t + dt)
    return normalize(m + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))


def llg(
    A, alpha, ms, m, dt, m_pad, f_n_demag, n, dx, h_zee=0.0, K=0.0, kaxis=(0, 0, 1)
):
    """Compute the effective field and perform one Euler LLG step."""
    h = h_eff(
        A=A, ms=ms, m=m, m_pad=m_pad, f_n_demag=f_n_demag, n=n, dx=dx, K=K, kaxis=kaxis
    )
    if not (np.isscalar(h_zee) and h_zee == 0.0):
        h += h_zee
    return pure_llg(alpha=alpha, gamma=GAMMA, m=m, dt=dt, h=h)


def llg_torque(
    A,
    alpha,
    K,
    kaxis,
    ms,
    m,
    dt,
    m_pad,
    f_n_demag,
    n,
    dx,
    h_zee=0.0,
    Hfl=0.0,
    Hdl=0.0,
    p=(0, 0, 0),
):
    h = h_eff(
        A=A, ms=ms, K=K, kaxis=kaxis, m=m, m_pad=m_pad, f_n_demag=f_n_demag, n=n, dx=dx
    )
    if not (np.isscalar(h_zee) and h_zee == 0.0):
        h += h_zee
    return pure_llg_torque(
        alpha=alpha, gamma=GAMMA, m=m, dt=dt, h=h, Hfl=Hfl, Hdl=Hdl, p=p
    )
