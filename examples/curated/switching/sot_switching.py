"""SOT current-induced switching hysteresis (cf. cmtj ``cims-hysteresis.py``), micromagnetic.

A perpendicular element (64 x 64 x 1 nm) on a heavy metal: current along x
gives a damping-like SOT with polarisation p = y. With an in-plane bias field
H_x the damping-like torque switches m_z deterministically, and the switching
polarity reverses with the sign of H_x. Each point of the current sweep is a
1 ns pulse (adaptive RK45) followed by relaxation with the current off; the
read-out is <m_z> (anomalous Hall signal). The damping-like field is converted
to a current density with a spin Hall angle theta_SH:
``Hdl = hbar theta_SH j / (2 e mu0 Ms d)``.
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
from micromagpy.torques import ELECTRON_CHARGE, HBAR  # noqa: E402

n = (16, 16, 1)
dx = (4e-9, 4e-9, 1e-9)
ms, A, K, damping = 8e5, 1.3e-11, 5e5, 0.1
theta_sh = 0.3
dt = 1e-12  # output interval (RK45)
HDL_PER_J = HBAR * theta_sh / (2 * ELECTRON_CHARGE * MU0 * ms * dx[2])


def sweep_points(hdl_max, points):
    """0 -> +max (initialise), then a closed loop +max -> -max -> +max."""
    init = np.linspace(0, hdl_max, points // 4 + 1)
    down = np.linspace(hdl_max, -hdl_max, points // 2 + 1)
    return np.concatenate([init, down[1:], down[::-1][1:]]), len(init)


def loop(field, hx, fl, sweep, pulse):
    """One current loop: per point a pulse (RK45) then relaxation with the current off; returns <m_z>, steps."""
    h_bias = jnp.array([1.0, 0.0, 0.0]) * hx
    m = mj.relax(jnp.zeros(n + (3,)).at[..., 2].set(1.0), field, ms, h_ext=h_bias)["m"]
    steps = round(pulse / dt)

    def point(m, hdl):
        res = mj.run(m, field, damping, dt, steps, h_ext=lambda t: h_bias, extra=mj.sot(fl * hdl, hdl, (0, 1, 0)),
                     solver="rk45", tol=1e-5, record_every=steps)
        m = mj.relax(res["m"], field, ms, h_ext=h_bias)["m"]
        return m, (m[..., 2].mean(), res["rk45_steps"])

    return jax.lax.scan(point, m, sweep)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=260)
    parser.add_argument("--hdl-max", type=float, default=2.0e5, help="max damping-like field (A/m)")
    parser.add_argument("--pulse", type=float, default=1e-9)
    args = parser.parse_args()
    field = mj.make_field(n, dx, ms, A, K=K)
    cases = [(+2e4, 0.0), (-2e4, 0.0), (+2e4, -1.2)]
    hdl, n_init = sweep_points(args.hdl_max, args.points)
    batch = jax.jit(jax.vmap(lambda hx, fl: loop(field, hx, fl, jnp.asarray(hdl), args.pulse)))  # all loops at once
    with timed("sot_switching", mesh=list(n), loops=len(cases), points_per_loop=len(hdl), solver="jax rk45 pulse + relax (vmap)",
               backend=f"jax float64 on {jax.devices()[0].device_kind}") as info:
        mz_all, steps = jax.block_until_ready(batch(*map(jnp.asarray, zip(*cases))))
        info["rk45_steps_total"] = int(steps.sum())
    results = []
    for (hx, fl), mz in zip(cases, np.asarray(mz_all)):
        results.append((hx, fl, hdl, mz, n_init))
        sw = hdl[np.where(np.diff(np.sign(mz)) != 0)[0] + 1]
        print(f"Hx = {hx:+.0e} A/m, Hfl/Hdl = {fl}: switching at Hdl = {np.round(sw / 1e3, 1)} kA/m")

    np.savez(fig_path("sot-switching.npz"), cases=np.array([c[:2] for c in cases]), hdl=results[0][2],
             j=results[0][2] / HDL_PER_J, mz=np.array([r[3] for r in results]), n_init=results[0][4])
    Hk_eff = 2 * K / (MU0 * ms) - ms
    fig, axes = plt.subplots(1, len(cases), figsize=(4.3 * len(cases), 3.8), sharey=True)
    for ax, (hx, fl, hdl, mz, n_init) in zip(axes, results):
        j = hdl / HDL_PER_J
        k = n_init - 1  # start of the loop at +max
        half = k + (len(hdl) - k) // 2
        ax.plot(j[k : half + 1] / 1e12, mz[k : half + 1], "o-", ms=3, color="crimson", label="+j -> -j")
        ax.plot(j[half:] / 1e12, mz[half:], "s-", ms=3, color="navy", label="-j -> +j")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(r"j ($10^{12}$ A/m$^2$)")
        title = f"$H_x$ = {hx / 1e3:+.0f} kA/m"
        if fl:
            title += f", $H_{{fl}}/H_{{dl}}$ = {fl}"
        ax.set_title(title)
        top = ax.secondary_xaxis("top", functions=(lambda x: x * 1e12 * HDL_PER_J / 1e3, lambda h: h * 1e3 / HDL_PER_J / 1e12))
        top.set_xlabel(r"$H_{dl}$ (kA/m)")
    axes[0].set_ylabel(r"$\langle m_z \rangle$ (AHE)")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        rf"SOT switching, 64x64x1 nm, $K_u$ = {K / 1e3:.0f} kJ/m$^3$ ($H_{{k,eff}}$ ~ {Hk_eff / 1e3:.0f} kA/m), "
        rf"$\theta_{{SH}}$ = {theta_sh}, 1 ns pulses",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fig_path("sot-switching.png"), dpi=200)


if __name__ == "__main__":
    main()
