"""Minimal time-stepping driver with support for thermal and 1/f noise sources."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from .config import DTYPE, GAMMA
from .demag import compute_demag_tensor, compute_trivial_tensor
from .fields import h_eff
from .llg import ll_operator, normalize, rk4, stochastic_heun
from .noise import ThermalField
from .drivers import as_driver
from .solvers import RK45State, integrate_rk45, llg_rhs_factory


def simulate(
    m: np.ndarray,
    dx: tuple,
    ms: float,
    A: float,
    alpha: float,
    dt: float,
    steps: int,
    K: float = 0.0,
    kaxis=(0, 0, 1),
    h_ext=0.0,
    demag=True,
    noise: Sequence = (),
    solver: str = "heun",
    record_every: int = 1,
    callback: Callable | None = None,
    progress: bool = False,
    tol: float = 1e-6,
    torques: Sequence = (),
):
    """Integrate the (stochastic) LLG equation.

    :param m: initial magnetisation ``n + (3,)``; cells with ``|m| = 0`` are
        treated as non-magnetic and ignored in averages.
    :param h_ext: external field (A/m): array, vector, callable ``h(t)`` or a
        :mod:`micromagpy.drivers` driver.
    :param demag: ``True`` (compute tensor), ``False`` (no demag) or a
        precomputed Fourier-space tensor.
    :param noise: noise sources. Objects with ``tick()`` and ``field(m)``
        (e.g. :class:`~micromagpy.noise.AnisotropyNoise`) are advanced once
        per step and their field is re-evaluated at the Heun predictor; any
        other source is called as ``source(t)`` once per step and its field
        held fixed (e.g. thermal or coloured field noise).
    :param solver: ``"heun"`` (2 field evaluations, Stratonovich; use with
        thermal noise), ``"rk4"`` (4 evaluations, deterministic or coloured
        noise only, most stable) or ``"euler"`` (1 evaluation; unstable for low
        damping and small cells, see :func:`micromagpy.llg.rk4`).
    :param torques: SOT / STT terms (:mod:`micromagpy.torques`), evaluated at
        every stage of every solver; amplitudes may be time dependent.
    :param callback: ``callback(i, t, m)`` called every ``record_every`` steps.
    :return: dict with ``t`` (s), ``m_avg`` (recorded averages) and ``m`` (final).
    """
    m = normalize(np.array(m, dtype=DTYPE))
    h_ext = as_driver(h_ext)
    n = m.shape[:3]
    magnetic = np.linalg.norm(m, axis=-1) > 0
    n_mag = max(int(magnetic.sum()), 1)
    if demag is True:
        f_n_demag = compute_demag_tensor(n, dx)
    elif demag is False or demag is None:
        f_n_demag = compute_trivial_tensor(n)
    else:
        f_n_demag = demag

    if solver not in ("heun", "rk4", "rk45", "euler"):
        raise ValueError("solver must be 'heun', 'rk4', 'rk45' or 'euler'")
    if solver in ("rk4", "rk45") and any(isinstance(s, ThermalField) for s in noise):
        raise ValueError("Use solver='heun' with thermal (white) noise")
    m_dependent = [s for s in noise if hasattr(s, "field") and hasattr(s, "tick")]
    fixed_sources = [s for s in noise if s not in m_dependent]

    def deterministic(mm):
        h = h_eff(A=A, ms=ms, K=K, kaxis=kaxis, m=mm, m_pad=None, f_n_demag=f_n_demag, n=n, dx=dx)
        for src in m_dependent:
            h += src.field(mm)
        return h

    iterator = range(steps)
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator)
        except ImportError:  # pragma: no cover
            pass

    rk_state = RK45State(dt=dt)
    extra = None
    if torques:

        def extra(mm, tt, out):
            for tq in torques:
                tq.add_rhs(mm, tt, alpha, GAMMA, out)
            return out

    times, m_avg = [], []
    for i in iterator:
        t = i * dt
        if i % record_every == 0:
            times.append(t)
            m_avg.append(m.sum(axis=(0, 1, 2)) / n_mag)
            if callback is not None:
                callback(i, t, m)
        for src in m_dependent:
            src.tick()
        h_noise = 0.0
        for src in fixed_sources:
            h_noise = h_noise + src(t)
        if solver in ("rk4", "rk45"):
            # external field evaluated at the RK stages, noise held over the step
            def total(mm, tt, h_noise=h_noise):
                h = deterministic(mm) + h_ext(tt)
                return h + h_noise

            if solver == "rk4":
                m = rk4(alpha, GAMMA, m, total, t, dt, extra_rhs=extra)
            else:
                rk_state.k1 = None  # the field function changes every interval
                rhs = llg_rhs_factory(alpha, GAMMA, total, extra_rhs=extra)
                m = integrate_rk45(rhs, m, t, t + dt, rk_state, tol)
            continue
        h_fixed = h_ext(t) + h_noise
        if solver == "heun":
            heun_extra = None if extra is None else (lambda mm, out, t=t: extra(mm, t, out))
            m = stochastic_heun(alpha, GAMMA, m, deterministic, h_fixed, dt, extra_rhs=heun_extra)
        elif solver == "euler":
            dm = ll_operator(alpha, GAMMA, m, deterministic(m) + h_fixed)
            if extra is not None:
                extra(m, t, dm)
            m = normalize(m + dt * dm)
    out = {"t": np.asarray(times), "m_avg": np.asarray(m_avg), "m": m}
    if solver == "rk45":
        out["rk45_steps"] = rk_state.accepted
        out["rk45_rejected"] = rk_state.rejected
    return out
