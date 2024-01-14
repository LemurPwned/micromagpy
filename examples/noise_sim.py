import time

import matplotlib.pyplot as plt
import numpy as np
from cmtj.noise import VectorAlphaNoise
from tqdm import tqdm

from micromagpy.config import DTYPE, GAMMA
from micromagpy.demag import compute_trivial_tensor
from micromagpy.fields import h_eff
from micromagpy.llg import euler_heun
from micromagpy.shapes import circular_mask, init_m, span_grid

# setup mesh and material constants
n = (100, 50, 1)
dx = (5e-9, 5e-9, 3e-9)
mu0 = 4e-7 * np.pi
ms = 8e5
A = 1.3e-11
alpha = 0.02
dt = 1e-13
np.random.seed(0)

if __name__ == "__main__":
    start = time.time()

    demag_start = time.time()

    m_pad = np.zeros([2 * i - 1 for i in n] + [3], dtype=DTYPE)
    n_blobs = 15
    grid = span_grid(n)
    m, m_pad = init_m(n)
    masks = []
    iters = 10000
    noise_sources = [
        VectorAlphaNoise(bufferSize=iters, std=0.1, alpha=2, scale=1e2)
        for _ in range(n_blobs)
    ]
    for _ in range(n_blobs):
        x0 = np.random.randint(0, n[0])
        y0 = np.random.randint(0, n[1])
        r1 = np.random.randint(1, 15)
        mask = circular_mask(x0, y0, r1, grid)
        m = np.logical_or(m, mask.astype(DTYPE) * np.random.choice([1.0, -1.0], 1))
        masks.append(mask)
    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(dpi=300)
        # z component, x component
        ax.pcolormesh(m[:, :, 0, -1].T)
        ax.set_xlabel("X (nm)")
        ax.set_ylabel("Y (nm)")
        fig.savefig("mask.png")
    m_pad[: n[0], : n[1], : n[2], :] = m
    # f_n_demag = compute_demag_tensor(n=n, dx=dx)
    f_n_demag = compute_trivial_tensor(n=n)
    demag_end = time.time()
    print("Demag setup took: ", demag_end - demag_start)
    # initialize magnetization that relaxes into s-state
    print(m.shape, masks[0].shape)
    for iteration in tqdm(range(iters), desc="Simulating"):
        m_pad[: n[0], : n[1], : n[2], :] = m
        hnoise = np.zeros(n + (3,), dtype=DTYPE)
        for i, source in enumerate(noise_sources):
            v = source.tickVector()
            # print(i, masks[i].shape)
            hnoise += masks[i].astype(DTYPE) * np.array([v.x, v.y, v.z])
        # generate the noise sources
        static_h_eff = h_eff(
            A=A,
            ms=ms,
            K=150e3,
            kaxis=(0, 0, 1),
            m=m.copy(),
            m_pad=m_pad,
            f_n_demag=f_n_demag,
            n=n,
            dx=dx,
        )
        m = euler_heun(
            m=m.copy(),
            h=static_h_eff,
            h_noise=hnoise,
            dt=dt,
            alpha=alpha,
            gamma=GAMMA,
        )
        if iteration % 50 == 0:
            with plt.style.context(["science", "nature"]):
                fig, ax = plt.subplots(dpi=300)
                # z component, x component
                ax.pcolormesh(m[:, :, 0, -1].T)
                ax.set_xlabel("X (nm)")
                ax.set_ylabel("Y (nm)")
                fig.savefig(f"./noise/m_{iteration}.png")
                plt.close(fig)
    quit()

    m[1:-1, :, :, 0] = 1.0
    m[(-1, 0), :, :, 1] = 1.0
