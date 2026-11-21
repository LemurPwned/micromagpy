"""Adaptive Runge-Kutta (Dormand-Prince 5(4)) integration of the LLG equation.

The step is accepted if the largest per-cell difference between the 5th and
4th order solutions, ``max |m5 - m4|``, is below ``tol``; the step size is then
adapted with the usual ``0.9 (tol / err)^(1/5)`` controller. ``m`` is
renormalised after every accepted step. The last stage is evaluated at the
renormalised solution and reused as the first stage of the next step (FSAL),
so an accepted step costs 6 field evaluations.

Not suitable for white (thermal) noise -- use the stochastic Heun scheme for that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit, prange  # noqa: F401

from .config import DTYPE
from .llg import _as_field, _ll_rhs, normalize

# Dormand-Prince coefficients
_C = (0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0)
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_B5 = (35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0)
_B4 = (5179 / 57600, 0.0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40)
_E = tuple(b5 - b4 for b5, b4 in zip(_B5, _B4))


@njit(parallel=True, cache=True)
def _damping_rhs(m, h, gamma, out):
    """Precession-free flow ``-gamma m x (m x h)`` (steepest descent on the sphere)."""
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                mx, my, mz = m[i, j, k, 0], m[i, j, k, 1], m[i, j, k, 2]
                hx, hy, hz = h[i, j, k, 0], h[i, j, k, 1], h[i, j, k, 2]
                cx = my * hz - mz * hy
                cy = mz * hx - mx * hz
                cz = mx * hy - my * hx
                out[i, j, k, 0] = -gamma * (my * cz - mz * cy)
                out[i, j, k, 1] = -gamma * (mz * cx - mx * cz)
                out[i, j, k, 2] = -gamma * (mx * cy - my * cx)


@njit(cache=True)
def _max_cell_norm(x):
    """max over cells of |x| (Euclidean norm of the 3-vector)."""
    nx, ny, nz, _ = x.shape
    best = 0.0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                v = x[i, j, k, 0] ** 2 + x[i, j, k, 1] ** 2 + x[i, j, k, 2] ** 2
                if v > best:
                    best = v
    return np.sqrt(best)


def llg_rhs_factory(alpha: float, gamma: float, field_fn):
    """``rhs(m, t)`` of the LLG equation for ``field_fn(m, t)`` (A/m)."""
    pref = gamma / (1 + alpha**2)

    def rhs(m, t):
        out = np.empty_like(m)
        _ll_rhs(m, _as_field(field_fn(m, t), m.shape), pref, alpha, out)
        return out

    return rhs


def damping_rhs_factory(gamma: float, field_fn):
    """``rhs(m, t)`` of the precession-free (energy minimising) flow."""

    def rhs(m, t):
        out = np.empty_like(m)
        _damping_rhs(m, _as_field(field_fn(m, t), m.shape), gamma, out)
        return out

    return rhs


@dataclass
class RK45State:
    """Carries the adaptive step size and the FSAL stage between calls."""

    dt: float
    k1: np.ndarray | None = None
    accepted: int = 0
    rejected: int = 0


def rk45_step(rhs, m: np.ndarray, t: float, state: RK45State, tol: float, dt_max: float = np.inf, dt_min: float = 1e-18):
    """Take one accepted Dormand-Prince step (retrying with smaller steps if needed).

    :return: ``(m_new, t_new, err)``; ``state.dt`` holds the proposed next step.
    """
    m = np.ascontiguousarray(m, dtype=DTYPE)
    k1 = rhs(m, t) if state.k1 is None else state.k1
    while True:
        h = min(state.dt, dt_max)
        ks = [k1]
        for s in range(1, 6):
            y = m.copy()
            for a, k in zip(_A[s], ks):
                if a != 0.0:
                    y += (h * a) * k
            ks.append(rhs(y, t + _C[s] * h))
        y5 = m.copy()
        for b, k in zip(_B5, ks):
            if b != 0.0:
                y5 += (h * b) * k
        y5 = normalize(y5)
        k7 = rhs(y5, t + h)
        ks.append(k7)
        err_vec = np.zeros_like(m)
        for e, k in zip(_E, ks):
            if e != 0.0:
                err_vec += (h * e) * k
        err = _max_cell_norm(err_vec)
        factor = 0.9 * (tol / err) ** 0.2 if err > 0 else 5.0
        factor = min(5.0, max(0.2, factor))
        if err <= tol or h <= dt_min:
            # a step clamped by dt_max keeps the previous proposal
            if h >= state.dt:
                state.dt = h * factor
            state.k1 = k7
            state.accepted += 1
            return y5, t + h, err
        state.rejected += 1
        state.dt = h * factor


def integrate_rk45(rhs, m: np.ndarray, t0: float, t1: float, state: RK45State, tol: float):
    """Integrate from ``t0`` to exactly ``t1`` with adaptive steps."""
    t = t0
    while t < t1 * (1 - 1e-14) and t1 - t > 1e-25:
        m, t, _ = rk45_step(rhs, m, t, state, tol, dt_max=t1 - t)
    return m
