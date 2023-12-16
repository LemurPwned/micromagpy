import numpy as np


def pure_llg(alpha: float, gamma: float, m: np.ndarray, dt: float, h: np.ndarray):
    mxh = np.cross(m, h)
    dmdt = -gamma / (1 + alpha**2) * mxh - alpha * gamma / (
        1 + alpha**2
    ) * np.cross(m, mxh)
    m += dt * dmdt
    return m / np.linalg.norm(m, axis=3, keepdims=True)


def compute_hdemag(
    m_pad: np.ndarray, f_n_demag: np.ndarray, n: tuple[int], axes: tuple
):
    f_m_pad = np.fft.fftn(m_pad, axes=axes)
    f_h_demag_pad = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
    f_h_demag_pad[:, :, :, 0] = (f_n_demag[:, :, :, (0, 1, 2)] * f_m_pad).sum(axis=3)
    f_h_demag_pad[:, :, :, 1] = (f_n_demag[:, :, :, (1, 3, 4)] * f_m_pad).sum(axis=3)
    f_h_demag_pad[:, :, :, 2] = (f_n_demag[:, :, :, (2, 4, 5)] * f_m_pad).sum(axis=3)
    return np.fft.ifftn(f_h_demag_pad, axes=axes)[: n[0], : n[1], : n[2], :].real


def compute_hex(A: float, mu0: float, ms: float, m: np.ndarray, dx: tuple, n: tuple):
    h_ex = -2 * m / dx[0] ** 2 - 2 * m / dx[1] ** 2 - 2 * m / dx[2] ** 2
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

    return 2 * A / (mu0 * ms) * h_ex


def h_eff(mu0, ms, m, m_pad, f_n_demag, n, dx):
    # demag field
    axes = tuple(filter(lambda i: n[i] > 1, range(3)))
    h_demag = compute_hdemag(m_pad, f_n_demag, axes)
    # exchange field
    h_ex = compute_hex(mu0, ms, m, dx, n)
    return ms * h_demag + h_ex
