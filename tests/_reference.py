"""Original (pre-optimisation) pure NumPy implementation, kept for regression tests."""

from math import pi

import numpy as np

from micromagpy.config import MU0
from micromagpy.demag import f, g


def set_n_demag(n, dx, n_demag, c, permute, func):
    it = np.nditer(n_demag[:, :, :, c], flags=["multi_index"], op_flags=["writeonly"])
    indx_iter = np.rollaxis(np.indices((2,) * 6), 0, 7).reshape(64, 6)
    dx_prod = np.prod(dx)
    while not it.finished:
        value = 0.0
        for i in indx_iter:
            idx = tuple(
                map(
                    lambda k: (it.multi_index[k] + n[k] - 1) % (2 * n[k] - 1)
                    - n[k]
                    + 1.0,
                    range(3),
                )
            )
            value += (-1.0) ** sum(i) * func(
                *map(lambda j: (idx[j] + i[j] - i[j + 3]) * dx[j], permute)
            )
        it[0] = -value / (4.0 * pi * dx_prod)
        it.iternext()
    return n_demag


def demag_tensor_real(n, dx):
    n_demag = np.zeros([2 * nk - 1 for nk in n] + [6])
    for i, t in enumerate(
        (
            (f, 0, 1, 2),
            (g, 0, 1, 2),
            (g, 0, 2, 1),
            (f, 1, 2, 0),
            (g, 1, 2, 0),
            (f, 2, 0, 1),
        )
    ):
        set_n_demag(n, dx, n_demag, i, t[1:], t[0])
    return n_demag


def hdemag(m, n_demag_real, n):
    axes = tuple(i for i in range(3) if n[i] > 1)
    f_n_demag = np.fft.fftn(n_demag_real, axes=axes)
    m_pad = np.zeros([2 * k - 1 for k in n] + [3])
    m_pad[: n[0], : n[1], : n[2], :] = m
    f_m_pad = np.fft.fftn(m_pad, axes=axes)
    out = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
    out[..., 0] = (f_n_demag[..., (0, 1, 2)] * f_m_pad).sum(axis=3)
    out[..., 1] = (f_n_demag[..., (1, 3, 4)] * f_m_pad).sum(axis=3)
    out[..., 2] = (f_n_demag[..., (2, 4, 5)] * f_m_pad).sum(axis=3)
    return np.fft.ifftn(out, axes=axes)[: n[0], : n[1], : n[2], :].real


def hex_field(A, ms, m, dx, n):
    h_ex = -2 * m * sum(1 / x**2 for x in dx)
    for i in range(6):
        h_ex += (
            np.repeat(
                m,
                (
                    1
                    if n[i % 3] == 1
                    else [(i // 3) * 2] + [1] * (n[i % 3] - 2) + [2 - (i // 3) * 2]
                ),
                axis=(i % 3),
            )
            / dx[i % 3] ** 2
        )
    return 2 * A / (MU0 * ms) * h_ex


def pure_llg(alpha, gamma, m, h, dt):
    mxh = np.cross(m, h)
    pref = gamma / (1 + alpha**2)
    m = m + dt * (-pref * mxh - alpha * pref * np.cross(m, mxh))
    return m / np.linalg.norm(m, axis=3, keepdims=True)
