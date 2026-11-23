"""Multilayer stacks: shaped layers, spacers with interlayer exchange coupling (IEC), on JAX.

Example (a synthetic antiferromagnet)::

    import jax.numpy as jnp
    from micromagpy.stack import Ellipse, Layer, Spacer, Stack

    pillar = Ellipse(64e-9, 64e-9)
    saf = Stack([Layer(pillar, 1.0e-9, ms=8e5, A=1.5e-11, K=9e5),
                 Spacer(0.8e-9, J1=-1e-3),
                 Layer(pillar, 1.2e-9, ms=9e5, A=1.5e-11, K=8e5)], cell=(4e-9, 4e-9))
    m = saf.relax(saf.uniform([(0, 0, 1), (0, 0, -1)]), h_ext=(0, 0, 5e4))["m"]
    out = saf.run(m, dt=1e-12, steps=5000, h_ext=lambda t: jnp.array([1e4, 0, 5e4]), solver="rk45")
    out["m_avg"]  # (records, layers, 3)

Every layer is one cell thick and all layers share the x/y grid (cell size
``cell``, extent from the largest shape, shapes centred). The magnetisation is
an array ``(layers, nx, ny, 3)``, zero outside each layer's shape.

Fields per layer: in-plane exchange (Neumann), uniaxial anisotropy, and the
demag field of every layer on every other one (Newell tensors for boxes of
different thickness, offset by the real gaps, see
:func:`micromagpy.demag.compute_layer_pair_tensors_real`); one FFT per layer
and step, so the cost grows with the number of layers, not its square. IEC
between the layers on either side of a spacer, cell by cell (cmtj convention):
``H_i = (J1 + 2 J2 m_i . m_j) / (mu0 Ms_i t_i) m_j``, energy per area
``-J1 m_i . m_j - J2 (m_i . m_j)^2`` (J1 < 0: antiparallel, J2 < 0: 90 degrees).

The constructor precomputes masks and tensors once; ``run`` / ``relax`` use
:func:`micromagpy.jax_backend.run` / ``relax``, so they jit, vmap and scan as
usual. ``J1`` / ``J2`` (and fields, torques, temperature) are arguments, so
sweeping them needs no recompilation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from . import jax_backend as mj
from .config import GAMMA, MU0
from .demag import compute_layer_pair_tensors_real
from .fft import padded_shape


@dataclass(frozen=True)
class Bar:
    """Rectangle ``lx x ly`` (m), centred."""

    lx: float
    ly: float

    def mask(self, x, y):
        return (np.abs(x) <= self.lx / 2) & (np.abs(y) <= self.ly / 2)


@dataclass(frozen=True)
class Ellipse:
    """Ellipse with axes ``lx x ly`` (full lengths, m), centred; ``lx == ly`` is a disk."""

    lx: float
    ly: float

    def mask(self, x, y):
        return (x / (self.lx / 2)) ** 2 + (y / (self.ly / 2)) ** 2 <= 1.0


@dataclass(frozen=True)
class Layer:
    """A magnetic layer: shape, thickness (m), Ms (A/m), A (J/m), Ku (J/m^3) along ``kaxis``, damping."""

    shape: Bar | Ellipse
    thickness: float
    ms: float
    A: float = 0.0
    K: float = 0.0
    kaxis: tuple = (0.0, 0.0, 1.0)
    alpha: float = 0.01


@dataclass(frozen=True)
class Spacer:
    """Non-magnetic spacer (m) with bilinear ``J1`` and biquadratic ``J2`` IEC (J/m^2)."""

    thickness: float
    J1: float = 0.0
    J2: float = 0.0


class Stack:
    """Layers separated by spacers: ``[Layer, Spacer, Layer, Spacer, Layer, ...]`` (bottom first)."""

    def __init__(self, items: Sequence[Layer | Spacer], cell: tuple, dtype=jnp.float64, interlayer_demag: bool = True):
        """``interlayer_demag=False`` keeps only each layer's own demag field (as in macrospin codes such as cmtj)."""
        items = list(items)
        if not items or not all(isinstance(it, Layer if k % 2 == 0 else Spacer) for k, it in enumerate(items)) or len(items) % 2 == 0:
            raise ValueError("A stack alternates Layer, Spacer, Layer, ... and starts and ends with a Layer")
        self.layers: tuple[Layer, ...] = tuple(items[0::2])
        self.spacers: tuple[Spacer, ...] = tuple(items[1::2])
        self.cell = (float(cell[0]), float(cell[1]))
        self.dtype = dtype
        L = len(self.layers)
        lx = max(layer.shape.lx for layer in self.layers)
        ly = max(layer.shape.ly for layer in self.layers)
        self.n = (max(1, round(lx / self.cell[0])), max(1, round(ly / self.cell[1])))
        x = (np.arange(self.n[0]) - (self.n[0] - 1) / 2) * self.cell[0]
        y = (np.arange(self.n[1]) - (self.n[1] - 1) / 2) * self.cell[1]
        X, Y = np.meshgrid(x, y, indexing="ij")
        self.masks = np.stack([layer.shape.mask(X, Y) for layer in self.layers])  # (L, nx, ny)
        if not self.masks.reshape(L, -1).any(1).all():
            raise ValueError("A layer has no cells; reduce the cell size")
        t = np.array([layer.thickness for layer in self.layers])
        z = np.concatenate([[0.0], np.cumsum([(t[k] + t[k + 1]) / 2 + s.thickness for k, s in enumerate(self.spacers)])])
        self.thickness, self.z = t, z
        self.ms = np.array([layer.ms for layer in self.layers])
        self.alpha = np.array([layer.alpha for layer in self.layers])

        # per-layer constants, shaped (L, 1, 1, 1) / (L, 1, 1, 3) to broadcast over (L, nx, ny, 3)
        col = lambda v: jnp.asarray(np.asarray(v, float).reshape(L, 1, 1, 1), dtype)  # noqa: E731
        self._cex = col([2 * layer.A / (MU0 * layer.ms) for layer in self.layers])
        self._hk = col([2 * layer.K / (MU0 * layer.ms) for layer in self.layers])
        u = np.array([np.asarray(layer.kaxis, float) / np.linalg.norm(layer.kaxis) for layer in self.layers])
        self._u = jnp.asarray(u.reshape(L, 1, 1, 3), dtype)
        self._iec_pref = col(1.0 / (MU0 * self.ms * t))
        self._alpha = col(self.alpha)
        self._mask = jnp.asarray(self.masks[..., None], dtype)
        self._counts = jnp.asarray(self.masks.sum((1, 2))[:, None], dtype)

        self._s = padded_shape(self.n + (1,))[:2]
        nr = compute_layer_pair_tensors_real(self.n, self.cell, t, z, shape=self._s)
        fn = np.fft.rfftn(nr, axes=(2, 3))  # (L, L, Px, Py', 6)
        sym = ((0, 1, 2), (1, 3, 4), (2, 4, 5))
        full = np.stack([np.stack([fn[..., sym[a][b]] for b in range(3)], -1) for a in range(3)], -2)
        if not interlayer_demag:
            full = full * np.eye(L)[:, :, None, None, None, None]
        self._fn = jnp.asarray(full * self.ms[None, :, None, None, None, None])  # Ms_j N_ij, (L, L, Px, Py', 3, 3)

    @property
    def n_layers(self) -> int:
        return len(self.layers)

    def uniform(self, directions) -> jnp.ndarray:
        """Initial state: one direction per layer (or one for all), zero outside each shape."""
        d = np.broadcast_to(np.asarray(directions, float), (self.n_layers, 3))
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return jnp.asarray(d[:, None, None, :] * self.masks[..., None], self.dtype)

    def layer_average(self, m) -> jnp.ndarray:
        """Average magnetisation of each layer over its cells, ``(layers, 3)``."""
        return m.sum((1, 2)) / self._counts

    def _couplings(self, J1, J2):
        k = len(self.spacers)
        J1 = [s.J1 for s in self.spacers] if J1 is None else J1
        J2 = [s.J2 for s in self.spacers] if J2 is None else J2
        J1, J2 = jnp.broadcast_to(jnp.asarray(J1), (k,)), jnp.broadcast_to(jnp.asarray(J2), (k,))
        return J1, J2

    def field(self, J1=None, J2=None) -> Callable:
        """``h(m)`` (A/m) for ``m`` of shape ``(layers, nx, ny, 3)``; ``J1`` / ``J2`` override the spacers' values."""
        J1, J2 = self._couplings(J1, J2)
        nx, ny = self.n
        ix2, iy2 = 1.0 / self.cell[0] ** 2, 1.0 / self.cell[1] ** 2

        def h(m):
            mp = jnp.pad(m, ((0, 0), (1, 1), (1, 1), (0, 0)), mode="edge")
            lap = (mp[:, 2:, 1:-1] + mp[:, :-2, 1:-1] - 2 * m) * ix2 + (mp[:, 1:-1, 2:] + mp[:, 1:-1, :-2] - 2 * m) * iy2
            out = self._cex * lap + self._hk * jnp.sum(m * self._u, -1, keepdims=True) * self._u
            fm = jnp.fft.rfftn(m, s=self._s, axes=(1, 2))
            hd = jnp.fft.irfftn(jnp.einsum("ijxyab,jxyb->ixya", self._fn, fm), s=self._s, axes=(1, 2))[:, :nx, :ny]
            out = out + hd.astype(m.dtype)
            if self.spacers:  # IEC between the layers on either side of spacer k
                lo, hi = m[:-1], m[1:]
                dot = jnp.sum(lo * hi, -1, keepdims=True)
                c = J1[:, None, None, None] + 2 * J2[:, None, None, None] * dot
                iec = jnp.zeros_like(m).at[:-1].add(c * hi * self._iec_pref[:-1]).at[1:].add(c * lo * self._iec_pref[1:])
                out = out + iec
            return out * self._mask

        return h

    def relax(self, m0, h_ext=0.0, J1=None, J2=None, **kwargs) -> dict:
        """Energy minimisation (:func:`micromagpy.jax_backend.relax`); torque criterion against the smallest Ms."""
        return mj.relax(m0, self.field(J1, J2), float(self.ms.min()), h_ext=h_ext, **kwargs)

    def thermal_sigma(self, temperature: float, dt: float) -> jnp.ndarray:
        """Per-layer thermal field std (A/m), shape ``(layers, 1, 1, 1)``."""
        s = [mj.thermal_sigma((*self.cell, layer.thickness), layer.ms, layer.alpha, temperature, dt) for layer in self.layers]
        return jnp.asarray(np.reshape(s, (-1, 1, 1, 1)), self.dtype)

    def run(self, m0, dt: float, steps: int, h_ext=None, J1=None, J2=None, temperature: float = 0.0, key=None,
            extra=None, **kwargs) -> dict:
        """Integrate LLG (:func:`micromagpy.jax_backend.run`) with per-layer damping.

        ``m_avg`` holds per-layer averages ``(records, layers, 3)``. ``extra`` torques
        (e.g. :func:`micromagpy.jax_backend.sot`) may take per-layer amplitudes of shape
        ``(layers, 1, 1, 1)``.
        """
        sigma = self.thermal_sigma(temperature, dt) if temperature else 0.0
        return mj.run(m0, self.field(J1, J2), self._alpha, dt, steps, h_ext=h_ext, sigma=sigma, key=key,
                      extra=extra, observe=self.layer_average, gamma=GAMMA, **kwargs)

    def energy_iec(self, m, J1=None, J2=None) -> jnp.ndarray:
        """IEC energy (J) per spacer, summed over the coupled cells."""
        J1, J2 = self._couplings(J1, J2)
        dot = jnp.sum(m[:-1] * m[1:], -1)
        area = self.cell[0] * self.cell[1]
        return area * jnp.sum(-J1[:, None, None] * dot - J2[:, None, None] * dot**2, (1, 2))
