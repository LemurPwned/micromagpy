# Copyright (C) 2014 Claas Abert
#
# This file is part of "70 lines of NumPy".
#
# "70 lines of NumPy" is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# "70 lines of NumPy" is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with magnum.fe. If not, see <http://www.gnu.org/licenses/>.

import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import asinh, atan, pi, sqrt

import numpy as np
from numba import jit
from tqdm import tqdm

DTYPE = np.float32
# setup mesh and material constants
n = (100, 25, 1)
dx = (5e-9, 5e-9, 3e-9)
mu0 = 4e-7 * pi
gamma = 2.211e5
ms = 8e5
A = 1.3e-11
alpha = 0.02

# a very small number
eps = 1e-18


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


# demag tensor setup
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


def compute_hdemag(m_pad: np.ndarray, f_n_demag: np.ndarray, axes: tuple):
    f_m_pad = np.fft.fftn(m_pad, axes=axes)
    f_h_demag_pad = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
    f_h_demag_pad[:, :, :, 0] = (f_n_demag[:, :, :, (0, 1, 2)] * f_m_pad).sum(axis=3)
    f_h_demag_pad[:, :, :, 1] = (f_n_demag[:, :, :, (1, 3, 4)] * f_m_pad).sum(axis=3)
    f_h_demag_pad[:, :, :, 2] = (f_n_demag[:, :, :, (2, 4, 5)] * f_m_pad).sum(axis=3)
    return np.fft.ifftn(f_h_demag_pad, axes=axes)[: n[0], : n[1], : n[2], :].real


def compute_hex(mu0: float, ms: float, m: np.ndarray, dx: tuple, n: tuple):
    # h_ex = -2 * m * sum(1 / x**2 for x in dx)
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


# compute effective field (demag + exchange)
def h_eff(m, m_pad, f_n_demag, n, dx):
    # demag field
    axes = tuple(filter(lambda i: n[i] > 1, range(3)))
    h_demag = compute_hdemag(m_pad, f_n_demag, axes)
    # exchange field
    h_ex = compute_hex(mu0, ms, m, dx, n)
    return ms * h_demag + h_ex


# compute llg step with optional zeeman field
def pure_llg(alpha, gamma, m, dt, h):
    mxh = np.cross(m, h)
    dmdt = -gamma / (1 + alpha**2) * mxh - alpha * gamma / (
        1 + alpha**2
    ) * np.cross(m, mxh)
    m += dt * dmdt
    return m / np.linalg.norm(m, axis=3, keepdims=True)


def llg(m, dt, m_pad, f_n_demag, n, dx, h_zee=0.0):
    m_pad[: n[0], : n[1], : n[2], :] = m
    h = h_eff(m, m_pad=m_pad, f_n_demag=f_n_demag, n=n, dx=dx) + h_zee
    return pure_llg(alpha, gamma, m, dt, h)


if __name__ == "__main__":
    start = time.time()

    demag_start = time.time()
    # setup demag tensor
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

    m_pad = np.zeros([2 * i - 1 for i in n] + [3], dtype=DTYPE)
    f_n_demag = np.fft.fftn(n_demag, axes=tuple(filter(lambda i: n[i] > 1, range(3))))
    demag_end = time.time()
    print("Demag setup took: ", demag_end - demag_start)
    # quit()
    # initialize magnetization that relaxes into s-state
    m = np.zeros(n + (3,), dtype=DTYPE)
    m[1:-1, :, :, 0] = 1.0
    m[(-1, 0), :, :, 1] = 1.0

    # relax
    alpha = 1.00
    for _ in tqdm(range(5000), desc="Relaxing"):
        llg(m, 2e-13, m_pad, f_n_demag, n, dx)
    # switch
    alpha = 0.02
    dt = 5e-15
    h_zee = np.tile([-24.6e-3 / mu0, +4.3e-3 / mu0, 0.0], np.prod(n)).reshape(m.shape)
    stime = 0.1e-9
    with open("sp4.dat", "w") as f:
        for i in tqdm(range(int(stime / dt))):
            f.write(
                "%f %f %f %f\n"
                % (
                    (i * 1e9 * dt,)
                    + tuple(map(lambda i: np.mean(m[:, :, :, i]), range(3)))
                )
            )
            llg(m, dt, m_pad, f_n_demag, n, dx, h_zee=h_zee)
    end = time.time()
    print("Took: ", end - start)
