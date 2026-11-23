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

Amplitudes may be scalars, per-cell arrays of shape ``n``, callables of time or
:mod:`micromagpy.drivers` (converted once with :func:`~micromagpy.drivers.as_driver`);
``p`` may be a 3-vector or a per-cell array ``n + (3,)`` (also as a driver). A
``mask`` restricts the torque to part of the mesh (e.g. the area under a
Hall bar or a nanopillar).
"""

from __future__ import annotations

import numpy as np

from .config import DTYPE, ELECTRON_CHARGE, HBAR, MU0  # noqa: F401  (constants re-exported)
from .drivers import as_driver


def _torque_kernel(m, p, afl, adl, pref, out):
    """out += -pref (afl m x p + adl m x (m x p))."""
    q = np.cross(m, p)
    out -= pref * (afl[..., None] * q + adl[..., None] * np.cross(m, q))


def _polarisation(p, n):
    p = np.asarray(p, dtype=DTYPE)
    if p.shape == (3,):
        norm = np.linalg.norm(p)
        return np.broadcast_to(p / norm if norm else p, tuple(n) + (3,))
    norm = np.linalg.norm(p, axis=-1, keepdims=True)
    return np.divide(p, norm, out=np.zeros(p.shape, dtype=DTYPE), where=norm > 0)


class _Torque:
    def __init__(self, p, mask=None):
        self.p = as_driver(p)
        self.mask = None if mask is None else np.asarray(mask, dtype=bool)
        self._p_cache = None  # normalised polarisation for a constant p, per mesh shape

    def unit_p(self, t: float, n: tuple) -> np.ndarray:
        """Unit polarisation broadcast to ``n + (3,)``."""
        if self.p.constant:
            if self._p_cache is None or self._p_cache.shape[:3] != tuple(n):
                self._p_cache = _polarisation(self.p(t), n)
            return self._p_cache
        return _polarisation(self.p(t), n)

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
        _torque_kernel(m, self.unit_p(t, n), afl, adl, gamma / (1 + alpha**2), out)
        return out

    def rhs(self, m, t, alpha, gamma):
        return self.add_rhs(m, t, alpha, gamma, np.zeros_like(m))


class SOT(_Torque):
    """Spin-orbit torque with field-like ``Hfl`` and damping-like ``Hdl`` amplitudes (A/m).

    :param Hfl, Hdl: scalar, per-cell array, callable ``f(t)`` or driver.
    :param p: spin polarisation (e.g. ``(0, 1, 0)`` for current along x on a
        heavy metal), vector, per-cell array, callable or driver.
    """

    def __init__(self, Hfl=0.0, Hdl=0.0, p=(0, 1, 0), mask=None):
        super().__init__(p, mask)
        self.Hfl = as_driver(Hfl)
        self.Hdl = as_driver(Hdl)

    def coefficients(self, m, t, alpha):
        hfl = np.asarray(self.Hfl(t), dtype=DTYPE)
        hdl = np.asarray(self.Hdl(t), dtype=DTYPE)
        return hfl - alpha * hdl, hdl + alpha * hfl


class STT(_Torque):
    """Slonczewski spin-transfer torque (cmtj convention).

    :param j: current density (A/m^2), scalar, per-cell array, callable ``j(t)`` or driver.
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
        self.j = as_driver(j)
        self.thickness = float(thickness)
        self.ms = float(ms)
        self.polarisation = float(polarisation)
        self.spacer_lambda = float(spacer_lambda)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self.alternative = bool(alternative)

    def a_j(self, t: float = 0.0):
        """Torque amplitude ``hbar j / (e mu0 Ms d)`` in A/m."""
        return HBAR * np.asarray(self.j(t), dtype=DTYPE) / (ELECTRON_CHARGE * MU0 * self.ms * self.thickness)

    def eta(self, m, t=0.0):
        p = self.unit_p(t, m.shape[:3])
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
