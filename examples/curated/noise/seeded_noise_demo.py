"""Seeded thermal + 1/f noise reproducibility (cf. cmtj ``seeded-noise-demo``).

A small PMA dot with thermal noise (100 K) and per-grain 1/f field noise is run
twice with the same seed and once with a different seed. Same-seed runs are
bit-identical; a different seed gives a different trajectory.
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, timed

from micromagpy.demag import compute_demag_tensor
from micromagpy.noise import ColouredNoiseField, ThermalField, voronoi_grains
from micromagpy.simulation import simulate

n = (16, 16, 1)
dx = (3e-9, 3e-9, 1e-9)
ms, A, K, damping, dt = 8e5, 1.3e-11, 6e5, 0.02, 1e-13
steps = 5000  # 0.5 ns
F_DEMAG = compute_demag_tensor(n, dx)


def run(seed: int):
    rng = np.random.default_rng(seed)
    m = np.zeros(n + (3,))
    m[..., 0] = 1.0  # start in-plane, like the cmtj demo
    noise = [
        ThermalField(n, dx, ms, damping, 100.0, dt, seed=rng),
        ColouredNoiseField(n, 5e3, 1.0, dt, regions=voronoi_grains(n, 8, seed=rng), seed=rng),
    ]
    res = simulate(m, dx, ms, A, damping, dt, steps, K=K, demag=F_DEMAG, noise=noise)
    print(f"seed {seed}: final <m> = {np.round(res['m_avg'][-1], 4)}")
    return res


def main():
    with timed("seeded_noise_demo", mesh=list(n), steps=3 * steps, solver="heun"):
        a, b, c = run(1224), run(1224), run(9876)
    assert np.array_equal(a["m_avg"], b["m_avg"]), "same seed must reproduce exactly"
    t = a["t"] * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    axes[0].plot(t, a["m_avg"][:, 2], color="navy", lw=2.0, label="Run 1")
    axes[0].plot(t, b["m_avg"][:, 2], color="crimson", lw=1.2, ls="--", label="Run 2")
    axes[0].set_title("Same seed")
    axes[1].plot(t, a["m_avg"][:, 2], color="navy", lw=2.0, label="Seed 1224")
    axes[1].plot(t, c["m_avg"][:, 2], color="darkgreen", lw=1.6, label="Seed 9876")
    axes[1].set_title("Different seeds")
    axes[0].set_ylabel(r"$\langle m_z \rangle$")
    for ax in axes:
        ax.set_xlabel("t (ns)")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
    fig.suptitle("Seeded thermal + 1/f noise reproducibility (16x16 mesh)")
    fig.tight_layout()
    fig.savefig(fig_path("seeded-noise-demo.png"), dpi=200)


if __name__ == "__main__":
    main()
