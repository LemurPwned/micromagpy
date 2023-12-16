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

import numpy as np
from tqdm import tqdm

from micromagpy.config import DTYPE
from micromagpy.demag import compute_demag_tensor
from micromagpy.llg import llg

# setup mesh and material constants
n = (100, 25, 1)
dx = (5e-9, 5e-9, 3e-9)
mu0 = 4e-7 * np.pi
ms = 8e5
A = 1.3e-11
alpha = 0.02


if __name__ == "__main__":
    start = time.time()

    demag_start = time.time()

    m_pad = np.zeros([2 * i - 1 for i in n] + [3], dtype=DTYPE)
    f_n_demag = compute_demag_tensor(n=n, dx=dx)

    demag_end = time.time()
    print("Demag setup took: ", demag_end - demag_start)
    # initialize magnetization that relaxes into s-state
    m = np.zeros(n + (3,), dtype=DTYPE)
    m[1:-1, :, :, 0] = 1.0
    m[(-1, 0), :, :, 1] = 1.0

    # relax
    alpha = 1.00
    for _ in tqdm(range(5000), desc="Relaxing"):
        llg(
            A=A,
            ms=ms,
            alpha=alpha,
            m=m,
            dt=2e-13,
            m_pad=m_pad,
            f_n_demag=f_n_demag,
            n=n,
            dx=dx,
        )
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
            llg(
                A=A,
                ms=ms,
                alpha=alpha,
                m=m,
                dt=2e-13,
                m_pad=m_pad,
                f_n_demag=f_n_demag,
                n=n,
                dx=dx,
                h_zee=h_zee,
            )
    end = time.time()
    print("Took: ", end - start)
