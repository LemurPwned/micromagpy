"""Minimal time-stepping driver with support for thermal and 1/f noise sources."""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np

from .config import DTYPE, GAMMA
from .demag import compute_demag_tensor, compute_trivial_tensor
from .fields import h_eff
from .llg import normalize, pure_llg, stochastic_heun


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
    callback: Optional[Callable] = None,
    progress: bool = False,
):
    """Integrate the (stochastic) LLG equation.

    :param m: initial magnetisation ``n + (3,)``; cells with ``|m| = 0`` are
        treated as non-magnetic and ignored in averages.
    :param h_ext: external field (A/m), array, vector or callable ``h(t)``.
    :param demag: ``True`` (compute tensor), ``False`` (no demag) or a
        precomputed Fourier-space tensor.
    :param noise: noise sources. Objects with ``tick()`` and ``field(m)``
        (e.g. :class:`~micromagpy.noise.AnisotropyNoise`) are advanced once
        per step and their field is re-evaluated at the Heun predictor; any
        other source is called as ``source(t)`` once per step and its field
        held fixed (e.g. thermal or coloured field noise).
    :param solver: ``"heun"`` (2 field evaluations, Stratonovich) or ``"euler"``.
    :param callback: ``callback(i, t, m)`` called every ``record_every`` steps.
    :return: dict with ``t`` (s), ``m_avg`` (recorded averages) and ``m`` (final).
    """
    m = normalize(np.array(m, dtype=DTYPE))
    n = m.shape[:3]
    magnetic = np.linalg.norm(m, axis=-1) > 0
    n_mag = max(int(magnetic.sum()), 1)
    if demag is True:
        f_n_demag = compute_demag_tensor(n, dx)
    elif demag is False or demag is None:
        f_n_demag = compute_trivial_tensor(n)
    else:
        f_n_demag = demag

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
        h_fixed = np.asarray(h_ext(t) if callable(h_ext) else h_ext, dtype=DTYPE)
        for src in fixed_sources:
            h_fixed = h_fixed + src(t)
        if solver == "heun":
            m = stochastic_heun(alpha, GAMMA, m, deterministic, h_fixed, dt)
        elif solver == "euler":
            m = pure_llg(alpha, GAMMA, m, deterministic(m) + h_fixed, dt)
        else:
            raise ValueError("solver must be 'heun' or 'euler'")
    return {"t": np.asarray(times), "m_avg": np.asarray(m_avg), "m": m}
