import numpy as np

from .config import MU0


def compute_anisotropy_field(ms: float, K: float, m: np.ndarray, axis: tuple[int]):
    Hk = (2 * K) / (MU0 * ms)
    kfield = np.ones_like(m) * axis
    return Hk * m * kfield


def compute_hex(A: float, ms: float, m: np.ndarray, dx: tuple, n: tuple):
    h_ex = -2 * m * sum(1 / x**2 for x in dx)
    for i in range(6):
        h_ex += (
            np.repeat(
                m,
                1
                if n[i % 3] == 1
                else [(i // 3) * 2] + [1] * (n[i % 3] - 2) + [2 - (i // 3) * 2],
                axis=(i % 3),
            )
            / dx[i % 3] ** 2
        )

    return 2 * A / (MU0 * ms) * h_ex


def compute_hdemag(
    m_pad: np.ndarray, f_n_demag: np.ndarray, n: tuple[int], axes: tuple
):
    f_m_pad = np.fft.fftn(m_pad, axes=axes)
    f_h_demag_pad = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
    f_h_demag_pad[:, :, :, 0] = (f_n_demag[:, :, :, (0, 1, 2)] * f_m_pad).sum(axis=3)
    f_h_demag_pad[:, :, :, 1] = (f_n_demag[:, :, :, (1, 3, 4)] * f_m_pad).sum(axis=3)
    f_h_demag_pad[:, :, :, 2] = (f_n_demag[:, :, :, (2, 4, 5)] * f_m_pad).sum(axis=3)
    return np.fft.ifftn(f_h_demag_pad, axes=axes)[: n[0], : n[1], : n[2], :].real


def h_eff(A, ms, K, kaxis, m, m_pad, f_n_demag, n, dx):
    # demag field
    axes = tuple(filter(lambda i: n[i] > 1, range(3)))
    # demag field
    h_demag = compute_hdemag(m_pad=m_pad, f_n_demag=f_n_demag, n=n, axes=axes)
    # exchange field
    h_ex = compute_hex(A=A, ms=ms, m=m, dx=dx, n=n)
    h_k = compute_anisotropy_field(ms=ms, K=K, m=m, axis=kaxis)
    return ms * h_demag + h_ex + h_k
