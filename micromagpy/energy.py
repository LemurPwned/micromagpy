"""Micromagnetic energies and energy minimisation (relaxation).

Energies (J) of the finite-difference discretisation, consistent with the
fields of :mod:`micromagpy.fields` (Neumann exchange, Newell demag):

* exchange    ``E_ex   = -mu0 Ms / 2 * V * sum(m . H_ex)``      (= A |grad m|^2)
* demag       ``E_d    = -mu0 Ms / 2 * V * sum(m . H_d)``
* anisotropy  ``E_K    = -K V sum((m . u)^2)``
* Zeeman      ``E_Z    = -mu0 Ms V sum(m . H_ext)``
"""

from __future__ import annotations

import numpy as np

from .config import DTYPE, GAMMA, MU0
from .demag import compute_demag_tensor, compute_trivial_tensor
from .fields import _unit_axis, compute_hdemag, compute_hex, h_eff
from .llg import normalize
from .solvers import RK45State, _max_cell_norm, damping_rhs_factory, rk45_step


def _tensor(n, dx, demag):
    if demag is True:
        return compute_demag_tensor(n, dx)
    if demag is False or demag is None:
        return compute_trivial_tensor(n)
    return demag


def energies(m, dx, ms, A=0.0, K=0.0, kaxis=(0, 0, 1), h_ext=0.0, demag=True) -> dict:
    """Energy terms (J) and their sum for magnetisation ``m``."""
    m = np.ascontiguousarray(m, dtype=DTYPE)
    n = m.shape[:3]
    V = float(np.prod(dx))
    f_n = _tensor(n, dx, demag)
    e = {}
    e["exchange"] = -0.5 * MU0 * ms * V * float(np.sum(m * compute_hex(A, ms, m, dx, n))) if A else 0.0
    e["demag"] = -0.5 * MU0 * ms * V * float(np.sum(m * ms * compute_hdemag(m, f_n, n)))
    e["anisotropy"] = -K * V * float(np.sum((m @ _unit_axis(kaxis)) ** 2)) if K else 0.0
    h_ext = np.broadcast_to(np.asarray(h_ext, dtype=DTYPE), m.shape)
    e["zeeman"] = -MU0 * ms * V * float(np.sum(m * h_ext))
    e["total"] = sum(e.values())
    return e


def relax(
    m,
    dx,
    ms,
    A=0.0,
    K=0.0,
    kaxis=(0, 0, 1),
    h_ext=0.0,
    demag=True,
    torque_tol: float = 1e-3,
    tol: float = 1e-4,
    max_steps: int = 100000,
    dt0: float = 1e-13,
    energy_every: int = 0,
):
    """Minimise the energy by integrating the precession-free flow ``dm/dt = -gamma m x (m x H)``.

    The flow is integrated with adaptive Dormand-Prince steps (``tol`` is the
    allowed per-cell change error, so large steps are taken once the dynamics
    slows down). Converged when ``max |m x H| / Ms < torque_tol``.

    :return: dict with ``m``, ``converged``, ``steps``, ``max_torque`` (A/m),
        ``energies`` (final, see :func:`energies`) and, with
        ``energy_every > 0``, an ``energy_history`` list of total energies.
    """
    m = normalize(np.array(m, dtype=DTYPE))
    n = m.shape[:3]
    f_n = _tensor(n, dx, demag)
    h_ext_arr = np.broadcast_to(np.asarray(h_ext, dtype=DTYPE), m.shape)

    def field(mm, t=0.0):
        return h_eff(A=A, ms=ms, K=K, kaxis=kaxis, m=mm, m_pad=None, f_n_demag=f_n, n=n, dx=dx) + h_ext_arr

    rhs = damping_rhs_factory(GAMMA, field)
    state = RK45State(dt=dt0)
    t = 0.0
    history = []
    torque = np.inf
    converged = False
    for step in range(max_steps):
        if energy_every and step % energy_every == 0:
            history.append(energies(m, dx, ms, A, K, kaxis, h_ext, f_n)["total"])
        m, t, _ = rk45_step(rhs, m, t, state, tol)
        # |dm/dt| = gamma |m x H| for the damping-only flow (|m| = 1)
        torque = _max_cell_norm(state.k1) / GAMMA
        if torque < torque_tol * ms:
            converged = True
            break
    return {
        "m": m,
        "converged": converged,
        "steps": state.accepted,
        "rejected": state.rejected,
        "pseudo_time": t,
        "max_torque": torque,
        "energies": energies(m, dx, ms, A, K, kaxis, h_ext, f_n),
        "energy_history": history,
    }
