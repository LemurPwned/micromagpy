"""Effective field contributions.

All fields are in A/m and act on arrays of shape ``n + (3,)``.
The heavy lifting is done by fused numba kernels which avoid the temporary
arrays the pure NumPy implementation allocated on every step.
"""

import numpy as np
from numba import njit, prange

from .config import DTYPE, MU0
from .fft import fft_axes, irfftn, padded_shape, rfftn


def _unit_axis(axis) -> np.ndarray:
    u = np.asarray(axis, dtype=DTYPE).reshape(3)
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("Anisotropy axis must be a non-zero vector")
    return u / norm


@njit(parallel=True, cache=True)
def _exchange_anisotropy_kernel(m, hd, scale_d, out, cex, idx2, hk, u, add_hd):
    """out = [scale_d * hd] + cex * laplacian(m) + hk * (m . u) u.

    Neumann (edge-replicated) boundaries; ``hd`` is only read if ``add_hd``.
    """
    nx, ny, nz, _ = m.shape
    for i in prange(nx):
        im = i - 1 if i > 0 else 0
        ip = i + 1 if i < nx - 1 else nx - 1
        for j in range(ny):
            jm = j - 1 if j > 0 else 0
            jp = j + 1 if j < ny - 1 else ny - 1
            for k in range(nz):
                km = k - 1 if k > 0 else 0
                kp = k + 1 if k < nz - 1 else nz - 1
                mu = m[i, j, k, 0] * u[0] + m[i, j, k, 1] * u[1] + m[i, j, k, 2] * u[2]
                for c in range(3):
                    mc = m[i, j, k, c]
                    lap = (
                        (m[ip, j, k, c] + m[im, j, k, c] - 2.0 * mc) * idx2[0]
                        + (m[i, jp, k, c] + m[i, jm, k, c] - 2.0 * mc) * idx2[1]
                        + (m[i, j, kp, c] + m[i, j, km, c] - 2.0 * mc) * idx2[2]
                    )
                    val = cex * lap + hk * mu * u[c]
                    if add_hd:
                        val += scale_d * hd[i, j, k, c]
                    out[i, j, k, c] = val


def compute_anisotropy_field(ms: float, K: float, m: np.ndarray, axis) -> np.ndarray:
    """Uniaxial anisotropy field ``2K/(mu0 Ms) (m . u) u``."""
    u = _unit_axis(axis)
    Hk = (2 * K) / (MU0 * ms)
    return Hk * np.einsum("...c,c->...", m, u)[..., None] * u


def compute_hex(
    A: float, ms: float, m: np.ndarray, dx: tuple, n: tuple = None
) -> np.ndarray:
    """Exchange field ``2A/(mu0 Ms) laplacian(m)`` with Neumann boundaries."""
    m = np.ascontiguousarray(m, dtype=DTYPE)
    out = np.empty_like(m)
    idx2 = np.array([1.0 / d**2 for d in dx], dtype=DTYPE)
    _exchange_anisotropy_kernel(
        m, m, 0.0, out, 2 * A / (MU0 * ms), idx2, 0.0, np.zeros(3, dtype=DTYPE), False
    )
    return out


@njit(parallel=True, cache=True)
def _tensor_multiply(fn, fm, out):
    """out = N(k) . m(k) for the symmetric tensor (xx, xy, xz, yy, yz, zz)."""
    a, b, c, _ = fm.shape
    for i in prange(a):
        for j in range(b):
            for k in range(c):
                mx = fm[i, j, k, 0]
                my = fm[i, j, k, 1]
                mz = fm[i, j, k, 2]
                out[i, j, k, 0] = (
                    fn[i, j, k, 0] * mx + fn[i, j, k, 1] * my + fn[i, j, k, 2] * mz
                )
                out[i, j, k, 1] = (
                    fn[i, j, k, 1] * mx + fn[i, j, k, 3] * my + fn[i, j, k, 4] * mz
                )
                out[i, j, k, 2] = (
                    fn[i, j, k, 2] * mx + fn[i, j, k, 4] * my + fn[i, j, k, 5] * mz
                )


def compute_hdemag(
    m_pad: np.ndarray, f_n_demag: np.ndarray, n: tuple, axes: tuple = None
) -> np.ndarray:
    """Demagnetisation field (in units of Ms) by FFT convolution.

    ``m_pad`` can either be the zero-padded magnetisation of shape
    ``(2n - 1) + (3,)`` or the unpadded magnetisation of shape ``n + (3,)``;
    in the latter case the padding is done implicitly by the FFT.
    """
    n = tuple(int(k) for k in n)
    if axes is None:
        axes = fft_axes(n)
    axes = tuple(axes)
    m_in = m_pad
    if m_in.shape[:3] != n:
        m_in = m_in[: n[0], : n[1], : n[2], :]
    if not axes:  # single cell: tensor is just a constant
        fm = m_in.astype(np.complex128)
        out = np.empty_like(fm)
        _tensor_multiply(f_n_demag, fm, out)
        return out.real.copy()
    pshape = padded_shape(n)
    s = [pshape[a] for a in axes]
    fm = rfftn(m_in, s=s, axes=axes)
    out = np.empty_like(fm)
    _tensor_multiply(f_n_demag, fm, out)
    h = irfftn(out, s=s, axes=axes)
    return h[: n[0], : n[1], : n[2], :]


def h_eff(A, ms, K, kaxis, m, m_pad, f_n_demag, n, dx):
    """Demag + exchange + uniaxial anisotropy field (A/m)."""
    n = tuple(int(k) for k in n)
    m = np.ascontiguousarray(m, dtype=DTYPE)
    hd = compute_hdemag(m_pad=m, f_n_demag=f_n_demag, n=n)
    h = np.empty_like(m)
    idx2 = np.array([1.0 / d**2 for d in dx], dtype=DTYPE)
    u = _unit_axis(kaxis) if K != 0 else np.zeros(3, dtype=DTYPE)
    _exchange_anisotropy_kernel(
        m, hd, float(ms), h, 2 * A / (MU0 * ms), idx2, (2 * K) / (MU0 * ms), u, True
    )
    return h
