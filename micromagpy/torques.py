"""Spin-orbit (SOT) and spin-transfer (STT) torques, following cmtj's conventions.

With ``gamma' = gamma0 / (1 + alpha^2)`` the LLG right-hand side becomes

``dm/dt = -gamma' [ m x H + alpha m x (m x H) + a_fl m x p + a_dl m x (m x p) ]``

* :class:`SOT` (cmtj ``Layer.createSOTLayer``): field-like ``Hfl`` and
  damping-like ``Hdl`` torque amplitudes in A/m,
  ``a_fl = Hfl - alpha Hdl`` and ``a_dl = Hdl + alpha Hfl``.
* :class:`STT` (cmtj ``Layer.createSTTLayer``, Slonczewski):
  ``aJ = hbar j / (e mu0 Ms d)``,
  ``eta = P L^2 / (L^2 + 1 + (L^2 - 1) m.p)`` (or ``P / (1 + L m.p)`` with
  ``alternative=True``), ``a_dl = kappa aJ eta`` and ``a_fl = -beta aJ eta``.

Amplitudes may be scalars, per-cell arrays of shape ``n`` or callables of time
(drivers); ``p`` may be a 3-vector or a per-cell array ``n + (3,)``. A
``mask`` restricts the torque to part of the mesh (e.g. the area under a
Hall bar or a nanopillar).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from .config import DTYPE, MU0

HBAR = 1.054571817e-34
ELECTRON_CHARGE = 1.602176634e-19


@njit(parallel=True, cache=True)
def _torque_kernel(m, p, afl, adl, pref, out):
    """out += -pref (afl m x p + adl m x (m x p))."""
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        for j in range(ny):
            for k in range(nz):
                mx, my, mz = m[i, j, k, 0], m[i, j, k, 1], m[i, j, k, 2]
                px, py, pz = p[i, j, k, 0], p[i, j, k, 1], p[i, j, k, 2]
                qx = my * pz - mz * py
                qy = mz * px - mx * pz
                qz = mx * py - my * px
                rx = my * qz - mz * qy
                ry = mz * qx - mx * qz
                rz = mx * qy - my * qx
                a, b = afl[i, j, k], adl[i, j, k]
                out[i, j, k, 0] -= pref * (a * qx + b * rx)
                out[i, j, k, 1] -= pref * (a * qy + b * ry)
                out[i, j, k, 2] -= pref * (a * qz + b * rz)


def _value(x, t):
    return x(t) if callable(x) else x


def _polarisation(p, n):
    p = np.asarray(p, dtype=DTYPE)
    if p.shape == (3,):
        norm = np.linalg.norm(p)
        return np.broadcast_to(p / norm if norm else p, tuple(n) + (3,))
    norm = np.linalg.norm(p, axis=-1, keepdims=True)
    return np.divide(p, norm, out=np.zeros(p.shape, dtype=DTYPE), where=norm > 0)


class _Torque:
    def __init__(self, p, mask=None):
        self.p = p
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)

    def coefficients(self, m, t, alpha):  # pragma: no cover - abstract
        raise NotImplementedError

    def add_rhs(self, m: np.ndarray, t: float, alpha: float, gamma: float, out: np.ndarray):
        """Add this torque's dm/dt to ``out`` (in place)."""
        n = m.shape[:3]
        afl, adl = self.coefficients(m, t, alpha)
        afl = np.broadcast_to(np.asarray(afl, dtype=DTYPE), n)
        adl = np.broadcast_to(np.asarray(adl, dtype=DTYPE), n)
        if self.mask is not None:
            mask = self.mask.any(axis=-1) if self.mask.shape == tuple(n) + (3,) else self.mask.reshape(n)
            afl, adl = afl * mask, adl * mask
        _torque_kernel(m, _polarisation(_value(self.p, t), n), afl, adl, gamma / (1 + alpha**2), out)
        return out

    def rhs(self, m, t, alpha, gamma):
        return self.add_rhs(m, t, alpha, gamma, np.zeros_like(m))


class SOT(_Torque):
    """Spin-orbit torque with field-like ``Hfl`` and damping-like ``Hdl`` amplitudes (A/m).

    :param Hfl, Hdl: scalar, per-cell array or callable ``f(t)``.
    :param p: spin polarisation (e.g. ``(0, 1, 0)`` for current along x on a
        heavy metal), vector, per-cell array or callable.
    """

    def __init__(self, Hfl=0.0, Hdl=0.0, p=(0, 1, 0), mask=None):
        super().__init__(p, mask)
        self.Hfl = Hfl
        self.Hdl = Hdl

    def coefficients(self, m, t, alpha):
        hfl = np.asarray(_value(self.Hfl, t), dtype=DTYPE)
        hdl = np.asarray(_value(self.Hdl, t), dtype=DTYPE)
        return hfl - alpha * hdl, hdl + alpha * hfl


class STT(_Torque):
    """Slonczewski spin-transfer torque (cmtj convention).

    :param j: current density (A/m^2), scalar, per-cell array or callable ``j(t)``.
    :param p: reference-layer magnetisation (polariser).
    :param thickness: free layer thickness d (m).
    :param ms: saturation magnetisation (A/m).
    :param polarisation: spin polarisation P.
    :param spacer_lambda: Slonczewski spacer parameter Lambda.
    :param beta: field-like / damping-like ratio.
    :param kappa: scaling of the damping-like term (cmtj ``setKappa``).
    :param alternative: simplified ``eta = P / (1 + Lambda m.p)`` (cmtj ``setAlternativeSTT``).
    """

    def __init__(
        self,
        j,
        p,
        thickness: float,
        ms: float,
        polarisation: float = 1.0,
        spacer_lambda: float = 1.0,
        beta: float = 0.0,
        kappa: float = 1.0,
        alternative: bool = False,
        mask=None,
    ):
        super().__init__(p, mask)
        self.j = j
        self.thickness = float(thickness)
        self.ms = float(ms)
        self.polarisation = float(polarisation)
        self.spacer_lambda = float(spacer_lambda)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.alternative = bool(alternative)

    def a_j(self, t: float = 0.0):
        """Torque amplitude ``hbar j / (e mu0 Ms d)`` in A/m."""
        return HBAR * np.asarray(_value(self.j, t), dtype=DTYPE) / (ELECTRON_CHARGE * MU0 * self.ms * self.thickness)

    def eta(self, m, t=0.0):
        p = _polarisation(_value(self.p, t), m.shape[:3])
        mp = np.einsum("...c,...c->...", m, p)
        lam = self.spacer_lambda
        if self.alternative:
            return self.polarisation / (1 + lam * mp)
        lam2 = lam * lam
        return self.polarisation * lam2 / (lam2 + 1 + (lam2 - 1) * mp)

    def coefficients(self, m, t, alpha):
        a = self.a_j(t) * self.eta(m, t)
        return -self.beta * a, self.kappa * a


def torque_rhs(m, t, alpha, gamma, torques, out=None):
    """Sum of the dm/dt contributions of ``torques`` (added to ``out`` if given)."""
    if out is None:
        out = np.zeros_like(m)
    for tq in torques:
        tq.add_rhs(m, t, alpha, gamma, out)
    return out
