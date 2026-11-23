"""Spin-transfer oscillator (cf. cmtj ``sto.py``), micromagnetic.

An in-plane free layer (64 x 64 x 2 nm) with a perpendicular polariser p = z
and Slonczewski STT (Lambda = 1, so eta = P/2). Above threshold the damping-like
torque balances damping on an out-of-plane precession orbit; in the macrospin
limit ``m_z = aJ eta / (alpha Ms)`` and the frequency is linear in current,
``f = gamma0 aJ P / (4 pi alpha)``. We sweep the current density, extract the
frequency from the PSD of <m_x> and compare with the analytic line, and show
the trajectory on the unit sphere for one current (cf. cmtj's figure).
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
from micromagpy.config import GAMMA  # noqa: E402
from micromagpy.torques import STT  # noqa: E402

n = (16, 16, 1)
dx = (4e-9, 4e-9, 2e-9)
ms, A, damping, P = 8e5, 1.3e-11, 0.02, 0.5
dt = 1e-12  # output interval (RK45)


def run(j, field, steps):
    m = jnp.zeros(n + (3,)).at[..., 0].set(1.0).at[..., 2].set(0.05)  # small seed tilt
    stt = mj.stt(j, (0, 0, 1), dx[2], ms, polarisation=P, spacer_lambda=1.0)
    return mj.run(m, field, damping, dt, steps, extra=stt, solver="rk45", tol=1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--currents", type=int, default=25)
    parser.add_argument("--time", type=float, default=6e-9)
    args = parser.parse_args()
    field = mj.make_field(n, dx, ms, A)
    js = np.linspace(1e10, 7e10, args.currents)
    steps = int(args.time / dt)
    batch = jax.jit(jax.vmap(lambda j: run(j, field, steps)))  # all currents at once
    freqs, mzs, mnorm, f_theory, example = [], [], [], [], None
    with timed("sto", mesh=list(n), currents=args.currents, time_per_current_ns=args.time * 1e9, solver="jax rk45 (vmap)",
               backend=f"jax float64 on {jax.devices()[0].device_kind}") as info:
        out = jax.block_until_ready(batch(jnp.asarray(js)))
        info["rk45_steps_total"] = int(out["rk45_steps"].sum())
    t_rec = np.asarray(out["t"][0])
    for j, m_avg in zip(js, np.asarray(out["m_avg"])):
        res = {"t": t_rec, "m_avg": m_avg}
        stt = STT(j, p=(0, 0, 1), thickness=dx[2], ms=ms, polarisation=P, spacer_lambda=1.0)
        tail = res["m_avg"][len(res["m_avg"]) // 2 :]
        # frequency from zero crossings of <m_x> (finer than the PSD bin width)
        x = tail[:, 0] - tail[:, 0].mean()
        crossings = np.count_nonzero(np.diff(np.sign(x)) != 0)
        freqs.append(crossings / (2 * len(x) * dt))
        mzs.append(tail[:, 2].mean())
        mnorm.append(np.linalg.norm(tail, axis=1).mean())
        f_theory.append(GAMMA * float(stt.a_j()) * P / (4 * np.pi * damping))
        print(f"j = {j:.2e} A/m^2: f = {freqs[-1] / 1e9:.2f} GHz (macrospin {f_theory[-1] / 1e9:.2f}), <mz> = {mzs[-1]:.3f}, |<m>| = {mnorm[-1]:.3f}")
        if example is None and j >= js[len(js) // 2]:
            example = (j, res)

    np.savez(fig_path("sto.npz"), j=js, f=freqs, mz=mzs, mnorm=mnorm, f_theory=f_theory,
             j_example=example[0], t_example=example[1]["t"], m_example=example[1]["m_avg"])
    fig = plt.figure(figsize=(13, 4.2))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    u, v = np.mgrid[0 : np.pi : 40j, 0 : 2 * np.pi : 40j]
    ax.plot_surface(np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u), alpha=0.15, color="lavender", lw=0)
    m = example[1]["m_avg"]
    ax.plot3D(m[:, 0], m[:, 1], m[:, 2], color="crimson", lw=0.8)
    ax.scatter([m[0, 0]], [m[0, 1]], [m[0, 2]], color="forestgreen", s=30, label="start")
    ax.set_axis_off()
    ax.set_title(f"<m> trajectory, j = {example[0]:.1e} A/m$^2$")
    ax.legend()
    ax2 = fig.add_subplot(1, 3, 2)
    t = example[1]["t"] * 1e9
    ax2.plot(t, m[:, 0], lw=0.6, label="<mx>")
    ax2.plot(t, m[:, 2], lw=1.2, label="<mz>")
    ax2.set_xlim(0, min(3, t[-1]))
    ax2.set_xlabel("t (ns)")
    ax2.set_title("<m> components")
    ax2.legend(loc="lower right")
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.plot(js / 1e10, np.array(freqs) / 1e9, "o", label="micromagnetic")
    ax3.plot(js / 1e10, np.array(f_theory) / 1e9, "k--", label=r"macrospin $\gamma_0 a_J P / 4\pi\alpha$")
    ax3.set_xlabel(r"j ($10^{10}$ A/m$^2$)")
    ax3.set_ylabel("f (GHz)")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.set_title("STO frequency vs current")
    ax3b = ax3.twinx()
    ax3b.plot(js / 1e10, mnorm, "s", color="gray", ms=4, label="|<m>| (uniformity)")
    ax3b.set_ylim(0, 1.6)
    ax3b.set_ylabel("|<m>|")
    ax3b.legend(fontsize=8, loc="center right")
    fig.tight_layout()
    fig.savefig(fig_path("sto.png"), dpi=200)


if __name__ == "__main__":
    main()
