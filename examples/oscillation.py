import time

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq
from tqdm import tqdm

from micromagpy.config import DTYPE
from micromagpy.demag import compute_demag_tensor
from micromagpy.llg import llg_torque
from micromagpy.shapes import circular_mask, init_m, span_grid


def simulation():
    n = (50, 50, 1)

    dx = (5e-9, 5e-9, 1.5e-9)

    grid = span_grid(n)
    _, m_pad = init_m(n)
    # here we define two junctions
    r1 = min(n[0], n[1]) // 2
    x0 = n[0] // 2
    y0 = n[1] // 2
    mask1 = circular_mask(x0, y0, r1, grid)
    m = mask1.astype(DTYPE) * 1.0
    m_pad[: n[0], : n[1], : n[2], :] = m
    f_n_demag = compute_demag_tensor(n=n, dx=dx)

    kaxis = np.array([0, 0, 1])
    K = 150e3
    ms = 8e5
    A = 1.3e-11
    alpha = 0.02
    p = (1.0, 0, 0)
    # relax
    Hfl = 1200
    alpha = 1.00
    for _ in tqdm(range(5000), desc="Relaxing"):
        if np.isnan(m).any():
            print("NaN detected")
            return
        m = llg_torque(
            A=A,
            K=K,
            kaxis=kaxis,
            ms=ms,
            alpha=alpha,
            m=m,
            dt=2e-13,
            m_pad=m_pad,
            f_n_demag=f_n_demag,
            n=n,
            dx=dx,
            p=p,
            Hfl=Hfl,
            Hdl=0.0,
        )
    alpha = 0.02
    dt = 1e-13
    h_zee = np.tile([0, 0, 15e3], np.prod(n)).reshape(m.shape)
    stime = 5e-9
    start = time.time()
    buffer = []
    print(m.shape)
    for i in tqdm(range(int(stime / dt))):
        if np.isnan(m).any():
            print("NaN detected")
            return
        buffer.append([i * 1e9 * dt] + [np.mean(m[..., i]) for i in range(3)])
        m = llg_torque(
            A=A,
            K=K,
            kaxis=kaxis,
            ms=ms,
            alpha=alpha,
            m=m,
            dt=1e-13,
            m_pad=m_pad,
            f_n_demag=f_n_demag,
            n=n,
            dx=dx,
            h_zee=h_zee,
            Hfl=Hfl,
            p=p,
            Hdl=0.0,
        )
        print(m)
    end = time.time()
    print("Took: ", end - start)
    buffer = np.asarray(buffer)
    y = fft(buffer[:, ..., -1])
    freqs = fftfreq(len(y), d=dt)
    y = y[: len(y) // 2]
    freqs = freqs[: len(freqs) // 2]
    findx = np.argwhere(freqs < 50e9)
    y = np.abs(y[findx])
    freqs = freqs[findx]

    np.savez("oscillation.npz", freqs=freqs, y=y)

    with plt.style.context(["science", "nature"]):
        fig, ax = plt.subplots(dpi=300)
        ax.plot(freqs / 1e9, 10 * np.log10(y))
        ax.set_xlabel("f (GHz)")
        ax.set_ylabel("Ampl. [dbm]")
        fig.savefig("oscillation.png")


if __name__ == "__main__":
    simulation()
