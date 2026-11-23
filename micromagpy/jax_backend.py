"""Optional JAX backend: the whole time loop is one compiled XLA program.

The numba path calls a handful of kernels from Python on every step, which is
fast on a CPU but leaves a GPU idle and pays the Python overhead per step.
Here the effective field, the LLG right-hand side and the solver step are
traced once and run inside :func:`jax.lax.scan`, on CPU or GPU, and
independent simulations (field / current sweeps, thermal ensembles) can be
batched with :func:`jax.vmap`. Install with ``pip install "micromagpy[jax]"``
(CPU) or ``pip install "jax[cuda13]"`` (NVIDIA GPU).

``dtype=jnp.float32`` is much faster on GPUs with little FP64 throughput
(consumer / GB10 class parts); keep the default float64 to match the numba
path to round-off.
The module sets ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` unless already set:
JAX otherwise grabs 75 % of device memory up front, which on GPUs that share
memory with the host (GB10, Jetson) starves the rest of the system.

Example::

    field = make_field(n, dx, ms, A, K=Ku, kaxis=(0, 0, 1))
    out = run(m0, field, alpha=0.01, dt=1e-13, steps=10_000, h_ext=lambda t: H)
    # 32 field points at once
    sweep = jax.vmap(lambda H: run(m0, field, 0.01, 1e-13, 10_000, h_ext=lambda t: H)["m_avg"])
"""

from __future__ import annotations

import os
from collections.abc import Callable

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")  # before the backend starts

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from .config import BOLTZMANN, GAMMA, MU0  # noqa: E402
from .solvers import _A, _B5, _C, _E  # noqa: E402  (Dormand-Prince tableau)
from .torques import ELECTRON_CHARGE, HBAR  # noqa: E402
from .demag import compute_demag_tensor_real  # noqa: E402
from .fft import fft_axes, padded_shape  # noqa: E402

jax.config.update("jax_enable_x64", True)


def make_field(n, dx, ms, A, K=0.0, kaxis=(0, 0, 1), demag=True, dtype=jnp.float64, D=0.0) -> Callable:
    """Return ``h(m)``: demag + exchange + uniaxial anisotropy + interfacial DMI (A/m), traceable by JAX.

    Same discretisation as :func:`micromagpy.fields.h_eff` (Newell tensor,
    Neumann exchange, bond DMI of :func:`micromagpy.fields.compute_hdmi`);
    ``K`` / ``kaxis`` may be per cell.
    """
    n = tuple(int(k) for k in n)
    axes = fft_axes(n)
    s = [padded_shape(n)[a] for a in axes]
    cex = 2 * A / (MU0 * ms)
    idx2 = [1.0 / d**2 for d in dx]
    hk = jnp.asarray(np.broadcast_to(np.asarray(K, float) * 2 / (MU0 * ms), n), dtype)
    u = np.broadcast_to(np.asarray(kaxis, float), n + (3,))
    norm = np.linalg.norm(u, axis=-1, keepdims=True)
    u = jnp.asarray(np.divide(u, norm, out=np.zeros(u.shape), where=norm > 0), dtype)
    if demag and axes:
        fn = jnp.fft.rfftn(jnp.asarray(compute_demag_tensor_real(n, dx), dtype), axes=axes)
    elif demag:  # single cell: tensor is a constant
        fn = jnp.asarray(compute_demag_tensor_real(n, dx), dtype)
    sym = ((0, 1, 2), (1, 3, 4), (2, 4, 5))

    def h(m):
        mp = jnp.pad(m, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
        lap = (
            (mp[2:, 1:-1, 1:-1] + mp[:-2, 1:-1, 1:-1] - 2 * m) * idx2[0]
            + (mp[1:-1, 2:, 1:-1] + mp[1:-1, :-2, 1:-1] - 2 * m) * idx2[1]
            + (mp[1:-1, 1:-1, 2:] + mp[1:-1, 1:-1, :-2] - 2 * m) * idx2[2]
        )
        out = cex * lap + (hk * jnp.sum(m * u, -1))[..., None] * u
        if D:  # missing neighbours (edges, empty cells) contribute nothing
            m0 = jnp.pad(m, ((1, 1), (1, 1), (0, 0), (0, 0)))
            gx, gy = m0[2:, 1:-1] - m0[:-2, 1:-1], m0[1:-1, 2:] - m0[1:-1, :-2]
            cx, cy = D / (MU0 * ms * dx[0]), D / (MU0 * ms * dx[1])
            out = out + jnp.stack([cx * gx[..., 2], cy * gy[..., 2], -cx * gx[..., 0] - cy * gy[..., 1]], -1)
        if not demag:
            return out
        if not axes:
            return out + ms * jnp.stack([sum(fn[..., r[j]] * m[..., j] for j in range(3)) for r in sym], -1)
        fm = jnp.fft.rfftn(m, s=s, axes=axes)
        fh = jnp.stack([sum(fn[..., r[j]] * fm[..., j] for j in range(3)) for r in sym], -1)
        hd = jnp.fft.irfftn(fh, s=s, axes=axes)[: n[0], : n[1], : n[2]]
        return out + ms * hd

    return h


def _as_fn(x):
    return x if callable(x) else (lambda t: x)


def _torque(p, coefficients) -> Callable:
    """``extra(m, t, alpha, pref) = -pref (afl m x p + adl m x (m x p))`` with ``(afl, adl) = coefficients(m, t, alpha)``."""
    p = jnp.asarray(p, float)
    p = p / jnp.linalg.norm(p, axis=-1, keepdims=True)

    def extra(m, t, alpha, pref):
        afl, adl = coefficients(m, t, alpha)
        q = jnp.cross(m, jnp.broadcast_to(p, m.shape))
        return -pref * (jnp.asarray(afl)[..., None] * q + jnp.asarray(adl)[..., None] * jnp.cross(m, q))

    return extra


def sot(Hfl, Hdl, p) -> Callable:
    """SOT term ``extra(m, t, alpha, pref)`` (cmtj convention, see :class:`micromagpy.torques.SOT`).

    ``Hfl`` / ``Hdl`` (A/m) may be numbers, per-cell arrays or traceable ``f(t)``.
    """
    fl, dl = _as_fn(Hfl), _as_fn(Hdl)

    def coefficients(m, t, alpha):
        a, b = fl(t), dl(t)
        return a - alpha * b, b + alpha * a

    return _torque(p, coefficients)


def stt(j, p, thickness, ms, polarisation=1.0, spacer_lambda=1.0, beta=0.0, kappa=1.0, alternative=False) -> Callable:
    """Slonczewski STT term ``extra(m, t, alpha, pref)`` (cmtj convention, see :class:`micromagpy.torques.STT`).

    ``j`` (A/m^2) may be a number, a per-cell array or a traceable ``j(t)``.
    """
    jf = _as_fn(j)
    pu = jnp.asarray(p, float)
    pu = pu / jnp.linalg.norm(pu, axis=-1, keepdims=True)
    lam2 = spacer_lambda**2

    def coefficients(m, t, alpha):
        a_j = HBAR * jf(t) / (ELECTRON_CHARGE * MU0 * ms * thickness)
        mp = jnp.sum(m * pu, -1)
        if alternative:
            eta = polarisation / (1 + spacer_lambda * mp)
        else:
            eta = polarisation * lam2 / (lam2 + 1 + (lam2 - 1) * mp)
        a = a_j * eta
        return -beta * a, kappa * a

    return _torque(p, coefficients)


def thermal_sigma(dx, ms, alpha, temperature, dt) -> float:
    """Per-component std of Brown's field (A/m), as :class:`micromagpy.noise.ThermalField`."""
    return float(np.sqrt(2 * alpha * BOLTZMANN * temperature / (GAMMA * MU0 * ms * np.prod(dx) * dt)))


def _normalize(m):
    norm = jnp.linalg.norm(m, axis=-1, keepdims=True)
    return jnp.where(norm > 0, m / jnp.where(norm > 0, norm, 1), m)


def _max_norm(x):
    return jnp.sqrt(jnp.max(jnp.sum(x * x, -1)))


def _dp45(rhs, m, t, h, k1):
    """One Dormand-Prince trial step: ``(y5, k7, err)`` as :func:`micromagpy.solvers.rk45_step`."""
    ks = [k1]
    for s in range(1, 6):
        ks.append(rhs(m + h * sum(a * k for a, k in zip(_A[s], ks) if a), t + _C[s] * h))
    y5 = _normalize(m + h * sum(b * k for b, k in zip(_B5, ks) if b))
    ks.append(rhs(y5, t + h))
    err = _max_norm(h * sum(e * k for e, k in zip(_E, ks) if e))
    return y5, ks[-1], err


def _dp45_attempt(rhs, m, t, dt, k1, tol, dt_max):
    """Try one step of size ``min(dt, dt_max)``; returns ``(accepted, m, t, k1, dt)`` (numba rules)."""
    h = jnp.minimum(dt, dt_max)
    y5, k7, err = _dp45(rhs, m, t, h, k1)
    factor = jnp.clip(jnp.where(err > 0, 0.9 * (tol / jnp.where(err > 0, err, 1.0)) ** 0.2, 5.0), 0.2, 5.0)
    ok = (err <= tol) | (h <= 1e-18)
    # an accepted step clamped by dt_max keeps the previous proposal
    new_dt = jnp.where(ok, jnp.where(h >= dt, h * factor, dt), h * factor)
    return ok, jnp.where(ok, y5, m), jnp.where(ok, t + h, t), jnp.where(ok, k7, k1), new_dt


def _integrate_rk45(rhs, m, t0, t1, dt, tol):
    """Adaptive Dormand-Prince from ``t0`` to exactly ``t1`` (as :func:`micromagpy.solvers.integrate_rk45`)."""

    def cond(s):
        _, t, _, _, _ = s
        return (t < t1 * (1 - 1e-14)) & (t1 - t > 1e-25)

    def body(s):
        m, t, k1, dt, n = s
        ok, m, t, k1, dt = _dp45_attempt(rhs, m, t, dt, k1, tol, t1 - t)
        return m, t, k1, dt, n + ok

    m, _, _, dt, n = jax.lax.while_loop(cond, body, (m, t0, rhs(m, t0), dt, 0))
    return m, dt, n


def run(
    m0,
    field: Callable,
    alpha: float,
    dt: float,
    steps: int,
    h_ext: Callable | None = None,
    extra: Callable | None = None,
    sigma: float = 0.0,
    key=None,
    solver: str = "heun",
    record_every: int = 1,
    gamma: float = GAMMA,
    observe: Callable | None = None,
    tol: float = 1e-6,
):
    """Integrate LLG for ``steps`` steps in one compiled loop.

    :param field: ``field(m)`` from :func:`make_field`.
    :param h_ext: traceable ``h_ext(t)`` (A/m), broadcastable to ``m``.
    :param extra: torque term ``extra(m, t, alpha, pref)`` (e.g. :func:`sot`).
    :param sigma: thermal field std (A/m, see :func:`thermal_sigma`); Heun only.
    :param solver: ``"heun"`` (stochastic Heun, Stratonovich), ``"rk4"`` or
        ``"rk45"`` (adaptive Dormand-Prince with error ``tol`` per cell; ``dt``
        is then only the output interval, as in :func:`micromagpy.simulation.simulate`).
    :param observe: ``observe(m)`` recorded instead of the average over magnetic
        cells (e.g. averages of ``m_x**2`` for magnetoresistance).
    :return: dict with ``t``, ``m_avg`` (every ``record_every`` steps, before
        the step as in :func:`micromagpy.simulation.simulate`), ``m`` and, for
        ``"rk45"``, ``rk45_steps`` (accepted steps).
    """
    if solver not in ("heun", "rk4", "rk45"):
        raise ValueError("solver must be 'heun', 'rk4' or 'rk45'")
    if sigma and solver != "heun":
        raise ValueError("Use solver='heun' with thermal noise")
    m0 = jnp.asarray(m0)
    m0 = _normalize(m0)
    mag = (jnp.linalg.norm(m0, axis=-1) > 0)[..., None]
    n_mag = jnp.maximum(mag.sum(), 1)
    pref = gamma / (1 + alpha**2)
    key = jax.random.PRNGKey(0) if key is None else key

    def rhs(m, t, h_fixed):
        h = field(m) + h_fixed
        if h_ext is not None:
            h = h + h_ext(t)
        c = jnp.cross(m, h)
        dm = -pref * (c + alpha * jnp.cross(m, c))
        if extra is not None:
            dm = dm + extra(m, t, alpha, pref)
        return dm.astype(m.dtype)  # keep float32 runs in float32

    def step(carry, i):
        m, key, hdt, nacc = carry
        t = (i * dt).astype(m0.dtype)
        if solver == "rk45":
            m, hdt, n = _integrate_rk45(lambda mm, tt: rhs(mm, tt, 0.0), m, t, t + dt, hdt, tol)
            return (m, key, hdt, nacc + n), None
        if solver == "rk4":
            k1 = rhs(m, t, 0.0)
            k2 = rhs(m + 0.5 * dt * k1, t + 0.5 * dt, 0.0)
            k3 = rhs(m + 0.5 * dt * k2, t + 0.5 * dt, 0.0)
            k4 = rhs(m + dt * k3, t + dt, 0.0)
            return (_normalize(m + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)), key, hdt, nacc), None
        hth = 0.0
        if sigma:
            key, sub = jax.random.split(key)
            hth = sigma * jax.random.normal(sub, m.shape, m.dtype) * mag
        k1 = rhs(m, t, hth)
        mp = _normalize(m + dt * k1)
        k2 = rhs(mp, t, hth)  # h_ext and torques held at t, as in simulate(solver="heun")
        return (_normalize(m + 0.5 * dt * (k1 + k2)), key, hdt, nacc), None

    def block(carry, j):
        m = carry[0]
        avg = (m * mag).sum((0, 1, 2)) / n_mag if observe is None else observe(m)
        carry, _ = jax.lax.scan(step, carry, j * record_every + jnp.arange(record_every))
        return carry, avg

    n_rec = steps // record_every
    carry = (m0, key, jnp.asarray(dt, m0.dtype), jnp.asarray(0))
    (m, _, _, nacc), m_avg = jax.lax.scan(block, carry, jnp.arange(n_rec))
    out = {"t": jnp.arange(n_rec) * record_every * dt, "m_avg": m_avg, "m": m}
    if solver == "rk45":
        out["rk45_steps"] = nacc
    return out


def relax(m0, field: Callable, ms: float, h_ext=0.0, torque_tol: float = 1e-3, tol: float = 1e-4,
          max_steps: int = 100000, dt0: float = 1e-13, gamma: float = GAMMA):
    """Minimise the energy with the precession-free flow ``dm/dt = -gamma m x (m x H)``.

    Same algorithm as :func:`micromagpy.energy.relax` (adaptive Dormand-Prince,
    converged when ``max |m x H| < torque_tol * Ms``), in one compiled loop, so it
    can be jitted, vmapped over fields or put in a ``lax.scan`` over a field sweep.
    ``tol`` bounds the attainable torque at about ``tol / (gamma dt)``: use
    ``tol`` ~ 1e-6 for tight ``torque_tol``.

    :param field: ``field(m)`` from :func:`make_field`.
    :param h_ext: constant external field (A/m), vector or ``n + (3,)`` array.
    :return: dict with ``m``, ``converged``, ``steps`` (accepted), ``rejected``,
        ``max_torque`` (A/m) and ``pseudo_time``.
    """
    m0 = _normalize(jnp.asarray(m0))
    h_ext = jnp.asarray(h_ext, m0.dtype)

    def rhs(m, t):
        h = field(m) + h_ext
        return (-gamma * jnp.cross(m, jnp.cross(m, h))).astype(m.dtype)

    def cond(s):
        _, _, _, _, acc, _, torque = s
        return (torque >= torque_tol * ms) & (acc < max_steps)

    def body(s):
        m, t, k1, dt, acc, rej, torque = s
        ok, m, t, k1, dt = _dp45_attempt(rhs, m, t, dt, k1, tol, jnp.inf)
        torque = jnp.where(ok, _max_norm(k1) / gamma, torque)
        return m, t, k1, dt, acc + ok, rej + (~ok), torque

    k1 = rhs(m0, 0.0)
    init = (m0, jnp.asarray(0.0, m0.dtype), k1, jnp.asarray(dt0, m0.dtype), 0, 0, jnp.asarray(jnp.inf, m0.dtype))
    m, t, _, _, acc, rej, torque = jax.lax.while_loop(cond, body, init)
    return {"m": m, "converged": torque < torque_tol * ms, "steps": acc, "rejected": rej,
            "max_torque": torque, "pseudo_time": t}
