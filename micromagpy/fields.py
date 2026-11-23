"""Effective field contributions.

All fields are in A/m and act on arrays of shape ``n + (3,)``.
Vectorised NumPy reference path; :func:`micromagpy.jax_backend.make_field`
is the same discretisation compiled with JAX.
"""

import numpy as np

from .config import DTYPE, MU0
from .fft import fft_axes, irfftn, padded_shape, rfftn


def _unit_axis(axis) -> np.ndarray:
    u = np.asarray(axis, dtype=DTYPE).reshape(3)
    norm = np.linalg.norm(u)
    if norm == 0:
        raise ValueError("Anisotropy axis must be a non-zero vector")
    return u / norm


def anisotropy_arrays(ms, K, kaxis, n):
    """Per-cell ``Hk = 2K/(mu0 Ms)`` of shape ``n`` and unit axes ``n + (3,)``.

    ``K`` may be a scalar or an array of shape ``n``; ``kaxis`` a 3-vector or
    an array of shape ``n + (3,)`` (normalised per cell; zero vectors allowed
    where ``K == 0``). Scalars are broadcast without copying.
    """
    n = tuple(int(k) for k in n)
    hk = np.asarray(K, dtype=DTYPE) * (2.0 / (MU0 * ms))
    hk = np.broadcast_to(hk, n)
    u = np.asarray(kaxis, dtype=DTYPE)
    if u.shape == (3,):
        u = np.broadcast_to(_unit_axis(u) if np.any(u) else u, n + (3,))
    else:
        u = np.broadcast_to(u, n + (3,))
        norm = np.linalg.norm(u, axis=-1, keepdims=True)
        u = np.divide(u, norm, out=np.zeros(n + (3,), dtype=DTYPE), where=norm > 0)
    return hk, u


def _exchange_anisotropy_kernel(m, hd, scale_d, out, cex, idx2, hk, u, add_hd):
    """out = [scale_d * hd] + cex * laplacian(m) + hk * (m . u) u.

    ``hk`` has shape n and ``u`` shape n + (3,) (per-cell uniaxial anisotropy).
    Neumann (edge-replicated) boundaries; ``hd`` is only read if ``add_hd``.
    """
    mp = np.pad(m, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
    out[...] = cex * (
        (mp[2:, 1:-1, 1:-1] + mp[:-2, 1:-1, 1:-1] - 2 * m) * idx2[0]
        + (mp[1:-1, 2:, 1:-1] + mp[1:-1, :-2, 1:-1] - 2 * m) * idx2[1]
        + (mp[1:-1, 1:-1, 2:] + mp[1:-1, 1:-1, :-2] - 2 * m) * idx2[2]
    ) + (hk * np.sum(m * u, -1))[..., None] * u
    if add_hd:
        out += scale_d * hd


def _dmi_kernel(m, out, cx, cy):
    """out += cx y x (m[i+1] - m[i-1]) - cy x x (m[j+1] - m[j-1]); missing neighbours are 0."""
    m0 = np.pad(m, ((1, 1), (1, 1), (0, 0), (0, 0)))
    gx, gy = m0[2:, 1:-1] - m0[:-2, 1:-1], m0[1:-1, 2:] - m0[1:-1, :-2]
    out[..., 0] += cx * gx[..., 2]
    out[..., 1] += cy * gy[..., 2]
    out[..., 2] -= cx * gx[..., 0] + cy * gy[..., 1]


def add_hdmi(out: np.ndarray, m: np.ndarray, D: float, ms: float, dx: tuple) -> np.ndarray:
    """Add the interfacial DMI field of :func:`compute_hdmi` to ``out`` in place."""
    _dmi_kernel(m, out, D / (MU0 * ms * dx[0]), D / (MU0 * ms * dx[1]))
    return out


def compute_hdmi(D: float, ms: float, m: np.ndarray, dx: tuple) -> np.ndarray:
    """Interfacial DMI field (A/m) for a film with normal z (MuMax3 ``Dind`` sign convention).

    Energy density ``D [m_z div m - (m . grad) m_z] = D [y . (m x dm/dx) - x . (m x dm/dy)]``,
    discretised on nearest-neighbour bonds, ``(D V / dx) y . (m_i x m_i+1)`` along x
    and ``-(D V / dy) x . (m_j x m_j+1)`` along y. Inside the magnet the field is
    the central difference ``2D/(mu0 Ms) (dmz/dx, dmz/dy, -dmx/dx - dmy/dy)``; at
    mesh edges and next to empty cells (``|m| = 0``) the missing bonds give the
    free-edge condition ``dm/dn = D/(2A) (z x n) x m`` in the continuum limit.
    The field is the exact gradient of :func:`micromagpy.energy.energies` ``"dmi"``.
    Resolve the DMI length ``2A/D`` with the cell size.
    """
    m = np.ascontiguousarray(m, dtype=DTYPE)
    return add_hdmi(np.zeros_like(m), m, D, ms, dx)


def compute_anisotropy_field(ms: float, K, m: np.ndarray, axis) -> np.ndarray:
    """Uniaxial anisotropy field ``2K/(mu0 Ms) (m . u) u`` (K, axis scalar/vector or per cell)."""
    hk, u = anisotropy_arrays(ms, K, axis, m.shape[:3])
    return (hk * np.einsum("...c,...c->...", m, u))[..., None] * u


def compute_hex(
    A: float, ms: float, m: np.ndarray, dx: tuple, n: tuple = None
) -> np.ndarray:
    """Exchange field ``2A/(mu0 Ms) laplacian(m)`` with Neumann boundaries."""
    m = np.ascontiguousarray(m, dtype=DTYPE)
    out = np.empty_like(m)
    idx2 = np.array([1.0 / d**2 for d in dx], dtype=DTYPE)
    hk, u = anisotropy_arrays(ms, 0.0, np.zeros(3), m.shape[:3])
    _exchange_anisotropy_kernel(m, m, 0.0, out, 2 * A / (MU0 * ms), idx2, hk, u, False)
    return out


def _tensor_multiply(fn, fm, out):
    """out = N(k) . m(k) for the symmetric tensor (xx, xy, xz, yy, yz, zz)."""
    for c, r in enumerate(((0, 1, 2), (1, 3, 4), (2, 4, 5))):
        out[..., c] = fn[..., r[0]] * fm[..., 0] + fn[..., r[1]] * fm[..., 1] + fn[..., r[2]] * fm[..., 2]


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


def h_eff(A, ms, K, kaxis, m, m_pad, f_n_demag, n, dx, D=0.0):
    """Demag + exchange + uniaxial anisotropy (+ interfacial DMI if ``D``) field (A/m).

    ``K`` / ``kaxis`` may be per cell (shapes ``n`` / ``n + (3,)``), e.g. for
    grains or domains with different anisotropy, see :func:`anisotropy_arrays`.
    """
    n = tuple(int(k) for k in n)
    m = np.ascontiguousarray(m, dtype=DTYPE)
    hd = compute_hdemag(m_pad=m, f_n_demag=f_n_demag, n=n)
    h = np.empty_like(m)
    idx2 = np.array([1.0 / d**2 for d in dx], dtype=DTYPE)
    hk, u = anisotropy_arrays(ms, K, kaxis, n)
    _exchange_anisotropy_kernel(m, hd, float(ms), h, 2 * A / (MU0 * ms), idx2, hk, u, True)
    if D:
        add_hdmi(h, m, D, ms, dx)
    return h
