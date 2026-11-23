"""Rxx and Rxy field sweeps of a PMA Hall bar (cf. cmtj ``harmonic-hall.py``).

Same material and resistance model as cmtj (Ms = 0.525 T, Ku = 154 kJ/m^3,
Kim's AMR / SMR / AHE formula), here a 128 x 128 nm, 1.45 nm thick element on
4 nm cells. The field, tilted 2 degrees from the plane at phi = 1 or 45
degrees, is swept -600 -> +600 -> -600 kA/m and every point is relaxed from
the previous state. Rxx follows m_x^2 / m_y^2 (AMR + SMR), Rxy follows m_z
(AHE) plus the planar Hall m_x m_y term at 45 degrees; both saturate at the
effective anisotropy field H_k,eff.
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, kim_resistance, timed  # noqa: E402

from micromagpy import jax_backend as mj  # noqa: E402
from micromagpy.config import MU0  # noqa: E402

n = (32, 32, 1)
dx = (4e-9, 4e-9, 1.45e-9)
ms, Ku, A = 0.525 / MU0, 1.54e5, 1.3e-11


FIELD = mj.make_field(n, dx, ms, A, K=Ku)


@jax.jit
def sweep(u, H):
    """Relax along the field sweep H * u; returns <m_x^2>, <m_y^2>, <m_x m_y>, <m_z> and relax steps per field."""

    def point(m, h):
        res = mj.relax(m, FIELD, ms, h_ext=h * u, torque_tol=1e-4, tol=1e-6)
        m = res["m"]
        obs = jnp.stack([(m[..., 0] ** 2).mean(), (m[..., 1] ** 2).mean(), (m[..., 0] * m[..., 1]).mean(), m[..., 2].mean()])
        return m, (obs, res["steps"])

    return jax.lax.scan(point, jnp.zeros(n + (3,)).at[..., 2].set(1.0), H)[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=int, default=100, help="points per sweep direction")
    args = parser.parse_args()
    up = np.linspace(-600e3, 600e3, args.points)
    H = np.concatenate([up, up[::-1]])
    out = {}
    with timed("magnetoresistance", mesh=list(n), points=2 * len(H), solver="jax relax",
               backend=f"jax float64 on {jax.devices()[0].device_kind}") as info:
        info["relax_steps_total"] = 0
        for phi in (1, 45):
            theta, ph = np.deg2rad(92), np.deg2rad(phi)
            u = jnp.array([np.sin(theta) * np.cos(ph), np.sin(theta) * np.sin(ph), np.cos(theta)])
            obs, s = jax.block_until_ready(sweep(u, jnp.asarray(H)))
            obs = np.asarray(obs)
            out[phi] = np.stack(kim_resistance((obs[:, 0], obs[:, 1]), obs[:, 2], obs[:, 3]), -1)
            info["relax_steps_total"] += int(s.sum())

    np.savez(fig_path("magnetoresistance.npz"), H=H, k=len(up), phi1=out[1], phi45=out[45])
    hk_eff = 2 * Ku / (MU0 * ms) - ms
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), sharex=True)
    k = len(up)
    for j, phi in enumerate(out):
        for i, name in enumerate(("$R_{xx}$", "$R_{xy}$")):
            ax = axes[i, j]
            ax.plot(H[:k] / 1e3, out[phi][:k, i], color="crimson", label="up")
            ax.plot(H[k:] / 1e3, out[phi][k:, i], "--", color="navy", label="down")
            for s in (-1, 1):
                ax.axvline(s * hk_eff / 1e3, color="gray", ls=":", lw=0.8)
            ax.set_ylabel(rf"{name} ($\Omega$)")
            ax.set_title(rf"$\phi$ = {phi}$^\circ$", fontsize=9)
        axes[1, j].set_xlabel("H (kA/m)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(rf"PMA Hall bar, $\theta$ = 92$^\circ$, dotted: $\pm H_{{k,eff}}$ (film) = {hk_eff / 1e3:.0f} kA/m", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path("magnetoresistance.png"), dpi=200)


if __name__ == "__main__":
    main()
