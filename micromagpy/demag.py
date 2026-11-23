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
node (vectorised NumPy) and apply the stencil with NumPy slicing, which is
equivalent up to floating point summation order but orders of magnitude faster.
"""

import numpy as np

from .config import DTYPE
from .fft import fft_axes, padded_shape, rfftn

pi = np.pi
_EPS = 1e-18


def f(x, y, z):
    x, y, z = np.abs(x), np.abs(y), np.abs(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    return (
        +y / 2.0 * (z**2 - x**2) * np.arcsinh(y / (np.sqrt(x**2 + z**2) + _EPS))
        + z / 2.0 * (y**2 - x**2) * np.arcsinh(z / (np.sqrt(x**2 + y**2) + _EPS))
        - x * y * z * np.arctan(y * z / (x * r + _EPS))
        + 1.0 / 6.0 * (2 * x**2 - y**2 - z**2) * r
    )


def g(x, y, z):
    z = np.abs(z)
    r = np.sqrt(x**2 + y**2 + z**2)
    return (
        +x * y * z * np.arcsinh(z / (np.sqrt(x**2 + y**2) + _EPS))
        + y / 6.0 * (3.0 * z**2 - y**2) * np.arcsinh(x / (np.sqrt(y**2 + z**2) + _EPS))
        + x / 6.0 * (3.0 * z**2 - x**2) * np.arcsinh(y / (np.sqrt(x**2 + z**2) + _EPS))
        - z**3 / 6.0 * np.arctan(x * y / (z * r + _EPS))
        - z * y**2 / 2.0 * np.arctan(x * z / (y * r + _EPS))
        - z * x**2 / 2.0 * np.arctan(y * z / (x * r + _EPS))
        - x * y * r / 3.0
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


def _node_values(use_g, n0, n1, n2, d0, d1, d2, p0, p1, p2):
    """Evaluate f or g on all nodes with offsets [-n_k, n_k] along each axis."""
    coord = np.meshgrid(*(np.arange(-k, k + 1) * d for k, d in ((n0, d0), (n1, d1), (n2, d2))), indexing="ij", sparse=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (g if use_g else f)(coord[p0], coord[p1], coord[p2])


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


def compute_layer_pair_tensors_real(n_xy: tuple, dx: tuple, thickness, z_center, shape: tuple = None) -> np.ndarray:
    """Real-space demag tensors between single-cell-thick layers on one x/y grid.

    Layer ``i`` is a sheet of cells ``dx[0] x dx[1] x thickness[i]`` centred at
    height ``z_center[i]``. ``N[i, j]`` gives the field averaged over a cell of
    layer ``i`` from a uniformly magnetised cell of layer ``j`` (offset target -
    source), on the padded x/y grid in FFT order, components (xx, xy, xz, yy, yz, zz).
    Along x/y the 64-term Newell sum is the usual second difference; along z it
    is the sum over the two corners of each box,
    ``sum_{u,v=+-1} u v F(dz + u t_i / 2 - v t_j / 2)``, which reduces to the
    second difference for equal thicknesses. ``N[i, i]`` is the single-layer tensor
    and ``V_i N[i, j] = V_j N[j, i]^T`` (reciprocity).
    """
    nx, ny = (int(k) for k in n_xy)
    thickness, z_center = np.asarray(thickness, float), np.asarray(z_center, float)
    L = len(thickness)
    scale = max(float(dx[0]), float(dx[1]), float(thickness.max()))
    d0, d1 = dx[0] / scale, dx[1] / scale
    t, zc = thickness / scale, z_center / scale
    if shape is None:
        shape = padded_shape((nx, ny, 1))[:2]
    out = np.zeros((L, L) + tuple(shape) + (6,), dtype=DTYPE)
    ix = np.ix_(_wrap_index(nx, shape[0]), _wrap_index(ny, shape[1]))
    xs = np.arange(-nx, nx + 1) * d0
    ys = np.arange(-ny, ny + 1) * d1
    uv = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    for i in range(L):
        for j in range(L):
            zs = np.array([zc[i] - zc[j] + u * t[i] / 2 - v * t[j] / 2 for u, v in uv])
            w = np.array([u * v for u, v in uv], float)
            coord = (xs[:, None, None], ys[None, :, None], zs[None, None, :])
            norm = -1.0 / (4.0 * pi * d0 * d1 * t[i])
            for c, (use_g, perm) in enumerate(_COMPONENTS):
                with np.errstate(divide="ignore", invalid="ignore"):
                    nodes = (g if use_g else f)(coord[perm[0]], coord[perm[1]], coord[perm[2]])
                nodes = _second_difference(_second_difference(nodes, 0), 1) @ w
                out[i, j, ..., c][ix] = nodes * norm
    return out
