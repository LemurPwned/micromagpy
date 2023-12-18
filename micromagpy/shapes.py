import numpy as np

from .config import DTYPE


def init_m(n: tuple[int]):
    m = np.zeros(n + (3,), dtype=DTYPE)
    m[1:-1, :, :, 0] = 1.0
    m[(-1, 0), :, :, 1] = 1.0
    m_pad = np.zeros([2 * i - 1 for i in n] + [3], dtype=DTYPE)
    return m, m_pad


def span_grid(n: tuple[int]):
    return np.mgrid[: n[0], : n[1], : n[2], :3]


def circular_mask(x0, y0, r, grid):
    xx, yy, _, _ = grid
    shape = (xx - x0) ** 2 + (yy - y0) ** 2
    return shape <= r**2


def elliptical_mask():
    ...
