import numpy as np

from .config import GAMMA, MU0


def compute_hdemag(
    m_pad: np.ndarray, f_n_demag: np.ndarray, n: tuple[int], axes: tuple
):
    f_m_pad = np.fft.fftn(m_pad, axes=axes)
    f_h_demag_pad = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
    f_h_demag_pad[:, :, :, 0] = (f_n_demag[:, :, :, (0, 1, 2)] * f_m_pad).sum(axis=3)
    f_h_demag_pad[:, :, :, 1] = (f_n_demag[:, :, :, (1, 3, 4)] * f_m_pad).sum(axis=3)
    f_h_demag_pad[:, :, :, 2] = (f_n_demag[:, :, :, (2, 4, 5)] * f_m_pad).sum(axis=3)
    return np.fft.ifftn(f_h_demag_pad, axes=axes)[: n[0], : n[1], : n[2], :].real


def pure_llg(alpha: float, gamma: float, m: np.ndarray, h, dt: float):
    mxh = np.cross(m, h)
    pref = gamma / (1 + alpha**2)
    dmdt = -pref * mxh - alpha * pref * np.cross(m, mxh)
    m += dt * dmdt
    return m / np.linalg.norm(m, axis=3, keepdims=True)


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


def h_eff(A, ms, m, m_pad, f_n_demag, n, dx):
    # demag field
    axes = tuple(filter(lambda i: n[i] > 1, range(3)))
    # demag field
    h_demag = compute_hdemag(m_pad=m_pad, f_n_demag=f_n_demag, n=n, axes=axes)
    # exchange field
    h_ex = compute_hex(A=A, ms=ms, m=m, dx=dx, n=n)
    return ms * h_demag + h_ex


def llg(A, alpha, ms, m, dt, m_pad, f_n_demag, n, dx, h_zee=0.0):
    m_pad[: n[0], : n[1], : n[2], :] = m
    h = h_eff(A=A, ms=ms, m=m, m_pad=m_pad, f_n_demag=f_n_demag, n=n, dx=dx) + h_zee
    return pure_llg(alpha=alpha, gamma=GAMMA, m=m.copy(), dt=dt, h=h)
