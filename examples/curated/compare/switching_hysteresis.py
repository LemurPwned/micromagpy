"""Field-driven hysteresis: cmtj macrospin vs micromagpy (cf. ``switching/switching_hysteresis.py``).

The macrospin has the disk's volume, Ms, Ku and demag factors, and follows
cmtj's ``switching-hysteresis.py`` protocol (10 ns of damped LLG at each of the
480 fields). The micromagnetic loop is read from
``switching/results/switching-hysteresis.npz``.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cmtj import AxialDriver, constantDriver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, load_micromag, macrospin, timed  # noqa: E402

from micromagpy.config import MU0  # noqa: E402
from micromagpy.shapes import circular_mask, span_grid  # noqa: E402

n, dx = (10, 10, 1), (3e-9, 3e-9, 2.1e-9)
ms, Ku = 1.03 / MU0, 8e3


def main():
    micro = load_micromag("switching", "switching-hysteresis")
    H, q = micro["H"], int(micro["q"])
    disk = circular_mask(4.5, 4.5, 5.0, span_grid(n))[..., 0]
    j = macrospin(n, dx, ms, Ku, kaxis=(1, 0, 0), damping=0.024, mask=disk, m0=(1, 0.1, 0.1))
    mx = []
    with timed("switching_hysteresis", points=len(H), solver="cmtj RK4, 10 ns per field"):
        for h in np.concatenate([[H.max()], H]):
            j.clearLog()
            j.setLayerExternalFieldDriver("free", AxialDriver(constantDriver(h), constantDriver(100.0), constantDriver(0)))
            j.runSimulation(10e-9, 1e-13, 10e-9)
            mx.append(j.getLayerMagnetisation("free").x)
    mx = np.array(mx[1:])

    hk = 2 * Ku / (MU0 * ms) / 1e3
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for ax, (label, m) in zip(axes, [("cmtj macrospin", mx), ("micromagpy, 10x10 cells (3 nm)", micro["mx"])]):
        up = slice(q, 3 * q)
        hc = [H[:q][np.argmax(m[:q] < 0)], H[up][np.argmax(m[up] > 0)]]
        ax.plot(H[:q] / 1e3, m[:q], color="crimson", lw=2, label=r"0 $\to$ $-H_{max}$")
        ax.plot(H[up] / 1e3, m[up], color="navy", lw=2, label=r"$-H_{max} \to +H_{max}$")
        ax.plot(H[3 * q :] / 1e3, m[3 * q :], "--", color="darkgreen", label=r"$+H_{max} \to 0$")
        for s in (-1, 1):
            ax.axvline(s * hk, color="gray", ls=":", label="$H_k$" if s > 0 else None)
        ax.set_title(f"{label}: $H_c$ = {hc[0] / 1e3:.1f} / {hc[1] / 1e3:+.1f} kA/m", fontsize=9)
        ax.set_xlabel("$H_x$ (kA/m)")
    axes[0].set_ylabel(r"$\langle m_x \rangle$")
    axes[0].legend(fontsize=7)
    fig.suptitle(rf"30 nm disk, 2.1 nm thick, $\mu_0 M_s$ = 1.03 T, $K_u$ = {Ku / 1e3:.0f} kJ/m$^3$ along x", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path("switching-hysteresis.png"), dpi=200)


if __name__ == "__main__":
    main()
