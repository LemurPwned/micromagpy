import time

import matplotlib.pyplot as plt
import numpy as np

from micromagpy.config import DTYPE
from micromagpy.demag import compute_demag_tensor
from micromagpy.llg import compute_hdemag
from micromagpy.shapes import circular_mask, init_m, span_grid


def simulation():
    n = (50, 50, 1)

    dx = (5e-9, 5e-9, 1.5e-9)

    grid = span_grid(n)
    _, m_pad = init_m(n)
    # here we define two junctions
    r1 = r2 = 5
    gap = 20
    x0 = x1 = n[0] // 2
    y0 = 10
    y1 = y0 + gap + r1 // 2 + r2 // 2  # 2 * r/2 because we measure centre to centre
    mask1 = circular_mask(x0, y0, r1, grid)
    mask2 = circular_mask(x1, y1, r2, grid)
    # we init to perpendicular anisotropy in both junctions
    m = (mask1 | mask2).astype(DTYPE) * 1.0
    m_pad[: n[0], : n[1], : n[2], :] = m
    demag_start = time.time()
    f_n_demag = compute_demag_tensor(n=n, dx=dx)
    demag_end = time.time()
    print(f"Demag computed in {demag_end-demag_start:.2f}")
    axes = list(filter(lambda i: n[i] > 1, range(3)))
    h_demag = compute_hdemag(m_pad=m_pad, f_n_demag=f_n_demag, n=n, axes=axes)
    hdemag_time = time.time()
    print(f"Field computed in {hdemag_time-demag_end:.2f}")
    hmag = np.linalg.norm(h_demag.squeeze(), axis=2)
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(dpi=300)
        ax.pcolormesh(
            np.arange(n[0]) * dx[0] * 1e9,
            np.arange(n[1]) * dx[1] * 1e9,
            np.log10(hmag).T,
        )
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        fig.savefig("demag.png")


if __name__ == "__main__":
    simulation()
