"""Optional JAX backend for the hot path (effective field + LLG time stepping).

Experimental. Mirrors :func:`micromagpy.fields.h_eff` (demag + exchange +
uniaxial anisotropy) and the Euler / Heun / RK4 steppers of
:mod:`micromagpy.llg`, written as pure functions so that a whole run can be
compiled with ``jax.jit`` and looped with ``jax.lax.scan`` (no Python
overhead per step). The numba path is untouched; this module is only usable
when ``jax`` is installed.

Precision: call :func:`enable_x64` (or set ``JAX_ENABLE_X64=1``) *before*
creating arrays to get float64; ``dtype=jnp.float32`` gives the single
precision variant (what GPUs are fast at).

Example::

    from micromagpy import jax_backend as jb
    jb.enable_x64()
    model = jb.JaxModel(n, dx, ms=8e5, A=1.3e-11, alpha=0.02)
    out = model.run(m0, dt=1e-13, steps=1000, solver="heun", h_ext=(0, 0, 1e4))
"""

from __future__ import annotations

from math import sqrt
from typing import Optional

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError as exc:  # pragma: no cover - optional dependency
    raise ImportError("micromagpy.jax_backend requires `pip install jax`") from exc

from .config import GAMMA, MU0
from .demag import compute_demag_tensor
from .fft import fft_axes, padded_shape
from .noise import BOLTZMANN

__all__ = [
    "enable_x64",
    "make_field_fn",
    "ll_rhs",
    "normalize",
    "euler_step",
    "heun_step",
    "rk4_step",
    "ou_coefficients",
    "ou_update",
    "thermal_sigma",
    "JaxModel",
]


def enable_x64(enable: bool = True):
    """Turn on float64 support in JAX (must be done before arrays are created)."""
    jax.config.update("jax_enable_x64", bool(enable))


# ---------------------------------------------------------------------------
# Field terms
# ---------------------------------------------------------------------------


def _demag_tensor(n, dx, f_n_demag, dtype, real_tensor=False):
    """Fourier-space tensor (6 components) as a JAX array of the right precision.

    The real-space demag tensor is even (diagonal) or odd-odd (off-diagonal)
    on the wrap-around grid, so its DFT is real up to round-off of the tensor
    evaluation (~1e-9 relative). With ``real_tensor=True`` only the real part
    is kept, which halves the tensor memory and the multiply cost; the default
    keeps the complex tensor for bit-level parity with the numba path.
    """
    if f_n_demag is None:
        f_n_demag = compute_demag_tensor(n, dx)
    f = np.asarray(f_n_demag)
    cdtype = jnp.complex64 if jnp.dtype(dtype) == jnp.float32 else jnp.complex128
    if real_tensor or not np.iscomplexobj(f):
        return jnp.asarray(f.real, dtype=dtype), True
    return jnp.asarray(f, dtype=cdtype), False


def _tensor_apply(fn, fm):
    """N(k) . m(k) for the symmetric tensor (xx, xy, xz, yy, yz, zz)."""
    mx, my, mz = fm[..., 0], fm[..., 1], fm[..., 2]
    hx = fn[..., 0] * mx + fn[..., 1] * my + fn[..., 2] * mz
    hy = fn[..., 1] * mx + fn[..., 3] * my + fn[..., 4] * mz
    hz = fn[..., 2] * mx + fn[..., 4] * my + fn[..., 5] * mz
    return jnp.stack([hx, hy, hz], axis=-1)


def _laplacian_neumann(m, idx2):
    """Seven-point Laplacian with edge-replicated (Neumann) boundaries.

    Axes with a single cell contribute nothing (as in the numba kernel).
    """
    lap = jnp.zeros_like(m)
    for ax in range(3):
        if m.shape[ax] == 1:
            continue
        first = jax.lax.slice_in_dim(m, 0, 1, axis=ax)
        last = jax.lax.slice_in_dim(m, m.shape[ax] - 1, m.shape[ax], axis=ax)
        inner_lo = jax.lax.slice_in_dim(m, 0, m.shape[ax] - 1, axis=ax)
        inner_hi = jax.lax.slice_in_dim(m, 1, m.shape[ax], axis=ax)
        plus = jnp.concatenate([inner_hi, last], axis=ax)  # m[i+1], clamped
        minus = jnp.concatenate([first, inner_lo], axis=ax)  # m[i-1], clamped
        lap = lap + (plus + minus - 2.0 * m) * idx2[ax]
    return lap


def make_field_fn(
    n, dx, ms, A=0.0, K=0.0, kaxis=(0, 0, 1), demag=True, f_n_demag=None, dtype=None, real_tensor=False
):
    """Return a pure, jit-able ``h(m)`` -> demag + exchange + anisotropy (A/m).

    :param demag: include the demagnetising field.
    :param f_n_demag: optional precomputed tensor from
        :func:`micromagpy.demag.compute_demag_tensor` (same padded grid).
    :param real_tensor: drop the (round-off) imaginary part of the tensor.
    """
    n = tuple(int(k) for k in n)
    if dtype is None:
        dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    axes = fft_axes(n)
    pshape = padded_shape(n)
    s = tuple(pshape[a] for a in axes)
    cex = 2.0 * A / (MU0 * ms)
    idx2 = tuple(1.0 / float(d) ** 2 for d in dx)
    if K != 0.0:
        u = np.asarray(kaxis, dtype=np.float64).reshape(3)
        u = u / np.linalg.norm(u)
        hk = 2.0 * K / (MU0 * ms)
    else:
        u, hk = np.zeros(3), 0.0
    u = jnp.asarray(u, dtype=dtype)
    fn = None
    if demag:
        fn, _ = _demag_tensor(n, dx, f_n_demag, dtype, real_tensor)
    ms_f = float(ms)

    def h_demag(m):
        if not axes:  # single cell: constant tensor
            return _tensor_apply(fn.reshape(6).real if jnp.iscomplexobj(fn) else fn.reshape(6), m)
        fm = jnp.fft.rfftn(m, s=s, axes=axes)
        hk_ = jnp.fft.irfftn(_tensor_apply(fn, fm), s=s, axes=axes)
        return hk_[: n[0], : n[1], : n[2], :].astype(m.dtype)

    def h_eff(m):
        h = jnp.zeros_like(m)
        if demag:
            h = h + ms_f * h_demag(m)
        if A != 0.0:
            h = h + cex * _laplacian_neumann(m, idx2)
        if hk != 0.0:
            h = h + hk * jnp.sum(m * u, axis=-1, keepdims=True) * u
        return h

    return h_eff


# ---------------------------------------------------------------------------
# LLG steppers (pure functions)
# ---------------------------------------------------------------------------


def _cross(a, b):
    ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
    bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
    return jnp.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], axis=-1)


def ll_rhs(m, h, alpha, gamma=GAMMA):
    """-pref (m x h) - alpha pref m x (m x h), pref = gamma / (1 + alpha^2)."""
    pref = gamma / (1.0 + alpha**2)
    mxh = _cross(m, h)
    return -pref * (mxh + alpha * _cross(m, mxh))


def normalize(m):
    """Unit-normalise each cell; cells with |m| = 0 stay zero."""
    norm = jnp.sqrt(jnp.sum(m * m, axis=-1, keepdims=True))
    return jnp.where(norm > 0, m / jnp.where(norm > 0, norm, 1.0), m)


def euler_step(field_fn, m, h_fixed, dt, alpha, gamma=GAMMA):
    return normalize(m + dt * ll_rhs(m, field_fn(m) + h_fixed, alpha, gamma))


def heun_step(field_fn, m, h_fixed, dt, alpha, gamma=GAMMA):
    """Same scheme as :func:`micromagpy.llg.stochastic_heun`."""
    k1 = ll_rhs(m, field_fn(m) + h_fixed, alpha, gamma)
    mp = normalize(m + dt * k1)
    k2 = ll_rhs(mp, field_fn(mp) + h_fixed, alpha, gamma)
    return normalize(m + 0.5 * dt * (k1 + k2))


def rk4_step(field_fn, m, h_fixed, dt, alpha, gamma=GAMMA):
    """Classical RK4 with ``h_fixed`` held over the step (as :func:`micromagpy.llg.rk4`)."""

    def f(mm):
        return ll_rhs(mm, field_fn(mm) + h_fixed, alpha, gamma)

    k1 = f(m)
    k2 = f(m + 0.5 * dt * k1)
    k3 = f(m + 0.5 * dt * k2)
    k4 = f(m + dt * k3)
    return normalize(m + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4))


def ou_coefficients(tau, sigma, dt):
    """Exact OU update ``x <- a x + b xi`` (as :class:`micromagpy.noise.OUAlphaNoise`)."""
    a = jnp.exp(-dt / tau)
    return a, sigma * jnp.sqrt(1.0 - a**2)


def ou_update(x, a, b, key):
    return x * a + b * jax.random.normal(key, x.shape, x.dtype)


_STEPPERS = {"euler": euler_step, "heun": heun_step, "rk4": rk4_step}


# ---------------------------------------------------------------------------
# Compiled model: single steps and scanned runs
# ---------------------------------------------------------------------------


class JaxModel:
    """Compiled field / step / run functions for a fixed mesh and material.

    :param thermal_T: temperature (K) for Brown's thermal field (``heun`` only),
        drawn with ``jax.random`` inside the scanned loop.
    :param ou: optional dict for a coloured (1/f^alpha) field built from a
        sum of Ornstein-Uhlenbeck processes, integrated exactly in the loop:
        ``dict(tau=array(P), sigma=array(P), direction=(3,) or None, shape=())``
        -- e.g. take ``tau``/``sigma`` from :class:`micromagpy.noise.OUAlphaNoise`.
        ``shape=()`` gives one process shared by all cells, ``shape=n`` one
        per cell. The field ``sum_k x_k`` is held fixed during a step.
    """

    def __init__(
        self,
        n,
        dx,
        ms,
        A=0.0,
        alpha=0.02,
        K=0.0,
        kaxis=(0, 0, 1),
        demag=True,
        f_n_demag=None,
        dtype=None,
        gamma=GAMMA,
        thermal_T: float = 0.0,
        ou: Optional[dict] = None,
        mask=None,
        real_tensor: bool = False,
    ):
        self.n = tuple(int(k) for k in n)
        self.dx = tuple(float(d) for d in dx)
        if dtype is None:
            dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        self.dtype = jnp.dtype(dtype)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ms = float(ms)
        self.field_fn = make_field_fn(self.n, self.dx, ms, A, K, kaxis, demag, f_n_demag, self.dtype, real_tensor)
        self.thermal_T = float(thermal_T)
        self.ou = ou
        self.mask = None if mask is None else jnp.asarray(np.asarray(mask, bool).reshape(self.n + (1,)), self.dtype)
        self.h_eff = jax.jit(self.field_fn)
        self.step = jax.jit(self._step, static_argnames=("solver",))
        self.run_scan = jax.jit(self._run, static_argnames=("steps", "solver", "record_every"))

    # -- single step -----------------------------------------------------
    def _step(self, m, h_fixed, dt, solver="euler"):
        return _STEPPERS[solver](self.field_fn, m, h_fixed, dt, self.alpha, self.gamma)

    # -- scanned run -----------------------------------------------------
    def _thermal_sigma(self, dt):
        vol = float(np.prod(self.dx))
        return jnp.sqrt(2 * self.alpha * BOLTZMANN * self.thermal_T / (self.gamma * MU0 * self.ms * vol * dt))

    def _run(self, m0, h_ext, dt, key, steps, solver="heun", record_every=1):
        """``steps`` steps in one compiled program; records <m> every ``record_every``."""
        stepper = _STEPPERS[solver]
        dtype = m0.dtype
        h_ext = jnp.broadcast_to(jnp.asarray(h_ext, dtype), m0.shape)
        magnetic = (jnp.sum(m0 * m0, axis=-1) > 0).astype(dtype)
        n_mag = jnp.maximum(magnetic.sum(), 1.0)
        thermal = self.thermal_T > 0.0
        sigma_th = self._thermal_sigma(dt).astype(dtype) if thermal else None

        ou = self.ou
        if ou is not None:
            tau = jnp.asarray(ou["tau"], dtype)
            sig = jnp.asarray(ou["sigma"], dtype)
            oshape = tuple(ou.get("shape", ()))
            direction = ou.get("direction")
            comp = () if direction is not None else (3,)
            bshape = (tau.size,) + (1,) * (len(oshape) + len(comp))
            a, b = ou_coefficients(tau, sig, dt)
            a, b = a.reshape(bshape), b.reshape(bshape)
            key, sub = jax.random.split(key)
            x0 = sig.reshape(bshape) * jax.random.normal(sub, (tau.size,) + oshape + comp, dtype)
            dvec = None if direction is None else jnp.asarray(np.asarray(direction) / np.linalg.norm(direction), dtype)
        else:
            x0 = jnp.zeros((), dtype)

        def ou_field(x):
            v = x.sum(axis=0)
            if dvec is not None:
                v = v[..., None] * dvec
            return jnp.broadcast_to(v, m0.shape)

        def one_step(carry, _):
            m, x, key = carry
            h = h_ext
            if ou is not None:
                key, sub = jax.random.split(key)
                x = ou_update(x, a, b, sub)
                h = h + ou_field(x)
            if thermal:
                key, sub = jax.random.split(key)
                hth = sigma_th * jax.random.normal(sub, m.shape, dtype)
                if self.mask is not None:
                    hth = hth * self.mask
                h = h + hth
            m = stepper(self.field_fn, m, h, dt, self.alpha, self.gamma)
            return (m, x, key), None

        def block(carry, _):
            avg = carry[0].sum(axis=(0, 1, 2)) / n_mag
            carry, _ = jax.lax.scan(one_step, carry, None, length=record_every)
            return carry, avg

        nrec = steps // record_every
        (m, _, _), avgs = jax.lax.scan(block, (m0, x0, key), None, length=nrec)
        return m, avgs

    def run(self, m0, dt, steps, solver="heun", h_ext=0.0, record_every=1, seed=0):
        """Convenience wrapper; returns NumPy ``{"t", "m_avg", "m"}`` like ``simulate``.

        ``steps`` must be a multiple of ``record_every``.
        """
        if steps % record_every:
            raise ValueError("steps must be a multiple of record_every")
        if self.thermal_T > 0 and solver != "heun":
            raise ValueError("Use solver='heun' with thermal (white) noise")
        m0 =jnp.asarray(m0, self.dtype)
        key = jax.random.PRNGKey(seed)
        m, avgs = self.run_scan(m0, jnp.asarray(h_ext, self.dtype), dt, key, steps=steps, solver=solver, record_every=record_every)
        t = np.arange(steps // record_every) * record_every * dt
        return {"t": t, "m_avg": np.asarray(avgs), "m": np.asarray(m)}


def thermal_sigma(dx, ms, alpha, temperature, dt, gamma=GAMMA) -> float:
    """Per-component std of Brown's field, identical to :class:`micromagpy.noise.ThermalField`."""
    return sqrt(2 * alpha * BOLTZMANN * temperature / (gamma * MU0 * ms * float(np.prod(dx)) * dt))
