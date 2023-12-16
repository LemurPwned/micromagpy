from concurrent.futures import ProcessPoolExecutor, as_completed
from math import asinh, atan, pi, sqrt

import numpy as np
from numba import jit

from .config import DTYPE


@jit(nopython=True, cache=True)
def f(x, y, z):
    eps = 1e-18
    x, y, z = abs(x), abs(y), abs(z)
    return (
        +y / 2.0 * (z**2 - x**2) * asinh(y / (sqrt(x**2 + z**2) + eps))
        + z / 2.0 * (y**2 - x**2) * asinh(z / (sqrt(x**2 + y**2) + eps))
        - x * y * z * atan(y * z / (x * sqrt(x**2 + y**2 + z**2) + eps))
        + 1.0 / 6.0 * (2 * x**2 - y**2 - z**2) * sqrt(x**2 + y**2 + z**2)
    )


@jit(nopython=True, cache=True)
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


def compute_demag_tensor(n: tuple[int], dx: tuple[float]):
    n_demag = np.zeros([2 * nk - 1 for nk in n] + [6], dtype=DTYPE)
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = []
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
            future = executor.submit(
                set_n_demag,
                n,
                dx,
                np.zeros([2 * nk - 1 for nk in n] + [6], dtype=DTYPE),
                i,
                t[1:],
                t[0],
            )
            futures.append(future)
        for future in as_completed(futures):
            n_demag += future.result()
    return np.fft.fftn(n_demag, axes=tuple(filter(lambda i: n[i] > 1, range(3))))
