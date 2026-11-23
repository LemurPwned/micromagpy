"""Field-driven hysteresis loop of an in-plane disk (cf. cmtj ``switching-hysteresis.py``).

cmtj's macrospin: Ms = 1.03 T, Ku = 8 kJ/m^3 along x, 2.1 nm thick, area of a
15 nm radius disk. Here the disk is resolved on 3 nm cells (full demag and
exchange). The field along the easy axis x is swept 0 -> -Hmax -> +Hmax -> 0
with a small transverse bias, and every point is relaxed with the energy
minimiser starting from the previous state (RK45 tol 1e-6: with the default
1e-4 the step-size error floor, ~tol / (gamma dt), stays above the torque
criterion and relax never converges), so the metastable branch (and the
coercive field) is kept. The Stoner-Wohlfarth coercive field of the macrospin,
H_k = 2 Ku / (mu0 Ms), is marked. The whole sweep is one ``lax.scan`` over the
JAX ``relax``.
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, timed  # noqa: E402

from micromagpy import jax_backend as mj  # noqa: E402
from micromagpy.config import MU0  # noqa: E402
from micromagpy.shapes import circular_mask, span_grid  # noqa: E402

n = (10, 10, 1)
dx = (3e-9, 3e-9, 2.1e-9)
ms = 1.03 / MU0
A, Ku, kaxis = 1.3e-11, 8e3, (1, 0, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=120, help="points per quarter loop")
    parser.add_argument("--hmax", type=float, default=70e3)
    args = parser.parse_args()
    disk = circular_mask(4.5, 4.5, 5.0, span_grid(n))[..., 0]
    m = np.zeros(n + (3,))
    m[disk] = (1.0, 0.1, 0.0)
    field = mj.make_field(n, dx, ms, A, K=Ku, kaxis=kaxis)
    q = args.points
    sweep = np.concatenate(
        [np.linspace(0, -args.hmax, q, endpoint=False), np.linspace(-args.hmax, args.hmax, 2 * q, endpoint=False),
         np.linspace(args.hmax, 0, q)]
    )
    def point(m, h):
        res = mj.relax(m, field, ms, h_ext=jnp.array([h, 100.0, 0.0]), torque_tol=1e-4, tol=1e-6)
        return res["m"], (res["m"][..., 0].sum() / disk.sum(), res["steps"])

    @jax.jit
    def loop(m):
        m = mj.relax(m, field, ms, h_ext=jnp.array([args.hmax, 100.0, 0.0]), tol=1e-6)["m"]
        return jax.lax.scan(point, m, jnp.asarray(sweep))[1]

    with timed("switching_hysteresis", mesh=list(n), points=len(sweep), solver="jax relax (RK45 damping flow)",
               backend=f"jax float64 on {jax.devices()[0].device_kind}") as info:
        mx, steps = jax.block_until_ready(loop(jnp.asarray(m)))
        info["relax_steps_total"] = int(steps.sum())
    mx = np.asarray(mx)
    up = slice(q, 3 * q)
    hc = [sweep[:q][np.argmax(mx[:q] < 0)], sweep[up][np.argmax(mx[up] > 0)]]
    print(f"coercive fields: {hc[0] / 1e3:.1f} / {hc[1] / 1e3:.1f} kA/m, H_k = {2 * Ku / (MU0 * ms) / 1e3:.1f} kA/m")

    np.savez(fig_path("switching-hysteresis.npz"), H=sweep, mx=mx, q=q, hc=hc)
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    ax.plot(sweep[:q] / 1e3, mx[:q], "-", color="crimson", lw=2, label=r"0 $\to$ $-H_{max}$")
    ax.plot(sweep[up] / 1e3, mx[up], "-", color="navy", lw=2, label=r"$-H_{max} \to +H_{max}$")
    ax.plot(sweep[3 * q :] / 1e3, mx[3 * q :], "--", color="darkgreen", label=r"$+H_{max} \to 0$")
    hk = 2 * Ku / (MU0 * ms) / 1e3
    for s in (-1, 1):
        ax.axvline(s * hk, color="gray", ls=":", label="Stoner-Wohlfarth $H_k$" if s > 0 else None)
    ax.set_xlabel("$H_x$ (kA/m)")
    ax.set_ylabel(r"$\langle m_x \rangle$")
    ax.set_title(
        rf"30 nm disk, $M_s$ = 1.03 T, $K_u$ = {Ku / 1e3:.0f} kJ/m$^3$: $H_c$ = {hc[0] / 1e3:.1f} / {hc[1] / 1e3:+.1f} kA/m",
        fontsize=9,
    )
    ax.legend(fontsize=8)
    ax.set_ylim(-1.1, 1.1)
    fig.tight_layout()
    fig.savefig(fig_path("switching-hysteresis.png"), dpi=200)


if __name__ == "__main__":
    main()
