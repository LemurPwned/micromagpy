"""Demagnetisation tensor (Newell et al. 1993) for a regular finite-difference mesh.

The tensor is evaluated on the zero-padded (2n - 1) grid in FFT ordering and
returned already transformed to Fourier space (real-to-complex transform over
the axes with more than one cell). Use it together with
:func:`micromagpy.fields.compute_hdemag`.

Implementation notes
--------------------
The original implementation evaluated, for every cell of the padded grid and
every tensor component, 64 terms of the Newell ``f``/``g`` functions in pure
Python. Each of those 64 terms is a second finite difference of ``f``/``g``
along every axis, i.e. a separable ``[-1, 2, -1]`` stencil applied to values of
the function on the (2n + 1) node grid. Here we evaluate ``f``/``g`` once per
node (numba, in parallel) and apply the stencil with NumPy slicing, which is
equivalent up to floating point summation order but orders of magnitude faster.
"""

from math import asinh, atan, pi, sqrt

import numpy as np
from numba import njit, prange

from .config import DTYPE
from .fft import fft_axes, padded_shape, rfftn


@njit(cache=True)
def f(x, y, z):
    eps = 1e-18
    x, y, z = abs(x), abs(y), abs(z)
    return (
        +y / 2.0 * (z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps))
        + z / 2.0 * (y**2 - x**2) * asinh(z / (sqrt(x**2 + y**2) + eps))
        - x * y * z * atan(y * z / (x * sqrt(x**2 + y**2 + z**2) + eps))
        + 1.0 / 6.0 * (2 * x**2 - y**2 - z**2) * sqrt(x**2 + y**2 + z**2)
    )


@njit(cache=True)
def g(x, y, z):
    eps = 1e-18
    z = abs(z)
    return (
        +x * y * z * asinh(z / (sqrt(x**2 + y**2) + eps))
        + y / 6.0 * (3.0 * z**2 - y**2) * asinh(x / (sqrt(y**2 + z**2) + eps))
        + x / 6.0 * (3.0 * z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps))
        - z**3 / 6.0 * atan(x * y / (z * sqrt(x**2 + y**2 + z**2) + eps))
        - z * y**2 / 2.0 * atan(x * z / (y * sqrt(x**2 + y**2 + z**2) + eps))
        - z * x**2 / 2.0 * atan(y * z / (x * sqrt(x**2 + y**2 + z**2) + eps))
        - x * y * sqrt(x**2 + y**2 + z**2) / 3.0
    )


# (use_g, permutation) for Nxx, Nxy, Nxz, Nyy, Nyz, Nzz
_COMPONENTS = (
    (False, (0, 1, 2)),
    (True, (0, 1, 2)),
    (True, (0, 2, 1)),
    (False, (1, 2, 0)),
    (True, (1, 2, 0)),
    (False, (2, 0, 1)),
)


@njit(parallel=True, cache=True)
def _node_values(use_g, n0, n1, n2, d0, d1, d2, p0, p1, p2):
    """Evaluate f or g on all nodes with offsets [-n_k, n_k] along each axis."""
    s0, s1, s2 = 2 * n0 + 1, 2 * n1 + 1, 2 * n2 + 1
    out = np.empty((s0, s1, s2), dtype=np.float64)
    for a in prange(s0):
        coord = np.empty(3)
        for b in range(s1):
            for c in range(s2):
                coord[0] = (a - n0) * d0
                coord[1] = (b - n1) * d1
                coord[2] = (c - n2) * d2
                if use_g:
                    out[a, b, c] = g(coord[p0], coord[p1], coord[p2])
                else:
                    out[a, b, c] = f(coord[p0], coord[p1], coord[p2])
    return out


def _second_difference(arr: np.ndarray, axis: int) -> np.ndarray:
    """Apply the [-1, 2, -1] stencil along ``axis`` (output shrinks by 2)."""
    lo = [slice(None)] * arr.ndim
    mid = [slice(None)] * arr.ndim
    hi = [slice(None)] * arr.ndim
    lo[axis], mid[axis], hi[axis] = slice(0, -2), slice(1, -1), slice(2, None)
    return 2.0 * arr[tuple(mid)] - arr[tuple(lo)] - arr[tuple(hi)]


def _wrap_index(nk: int, size: int) -> np.ndarray:
    """Positions of offsets -(nk-1)..(nk-1) on a periodic grid of ``size``."""
    return np.arange(-(nk - 1), nk) % size


def compute_demag_tensor_real(n: tuple, dx: tuple, shape: tuple = None) -> np.ndarray:
    """Real-space demag tensor on the padded grid, FFT (wrap-around) ordered.

    ``shape`` defaults to :func:`micromagpy.fft.padded_shape` and must be at
    least ``2n - 1`` along every axis with more than one cell.
    Components are ordered (xx, xy, xz, yy, yz, zz).
    """
    n = tuple(int(k) for k in n)
    # N is dimensionless, so evaluate f/g in units of the largest cell edge.
    # This keeps the regularisation eps = 1e-18 in f/g negligible; in SI units
    # (dx ~ 1e-9 m) terms such as x * r ~ 1e-27 are swamped by eps, which
    # biases the self-demag factors by ~10%.
    scale = max(float(k) for k in dx)
    dx = tuple(float(k) / scale for k in dx)
    if shape is None:
        shape = padded_shape(n)
    if any(s < 2 * k - 1 for s, k in zip(shape, n)):
        raise ValueError(f"Padded shape {shape} too small for mesh {n}")
    n_demag = np.zeros(tuple(shape) + (6,), dtype=DTYPE)
    ix = np.ix_(*(_wrap_index(nk, s) for nk, s in zip(n, shape)))
    norm = -1.0 / (4.0 * pi * np.prod(dx))
    for c, (use_g, perm) in enumerate(_COMPONENTS):
        nodes = _node_values(use_g, *n, *dx, *perm)
        for axis in range(3):
            nodes = _second_difference(nodes, axis)
        # nodes now holds offsets [-(n-1), n-1] in natural order
        n_demag[..., c][ix] = nodes * norm
    return n_demag


def compute_demag_tensor(n: tuple, dx: tuple) -> np.ndarray:
    """Fourier-transformed demag tensor, ready for :func:`compute_hdemag`."""
    n_demag = compute_demag_tensor_real(n, dx)
    axes = fft_axes(n)
    if not axes:  # single cell
        return n_demag.astype(np.complex128)
    return np.ascontiguousarray(rfftn(n_demag, axes=axes))


def compute_trivial_tensor(n: tuple) -> np.ndarray:
    """All-zero (Fourier space) tensor, i.e. demagnetisation switched off."""
    axes = fft_axes(n)
    shape = list(padded_shape(n))
    if axes:
        shape[axes[-1]] = shape[axes[-1]] // 2 + 1
    return np.zeros(shape + [6], dtype=np.complex128)
