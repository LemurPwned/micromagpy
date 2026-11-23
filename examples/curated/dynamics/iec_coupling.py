"""Coupled oscillations of two IEC-coupled layers (cf. cmtj ``iec-coupling.py``).

Two perpendicular layers (Ms = 1.0 / 1.2 T, Ku = 300 / 800 kJ/m^3, 1.4 / 3 nm,
alpha = 0.011) are coupled antiferromagnetically through a 1 nm spacer,
J1 = -4e-5 J/m^2, and driven by a field along y, 100 kA/m + 50 kA/m sin(2 pi 5 GHz t),
for 30 ns, starting antiparallel (up / down). Here each layer is a 48 nm disk on
3 nm cells (:class:`micromagpy.stack.Stack`), with the demag fields of both layers
on each other; cmtj's macrospin uses a thin-film demag tensor. The figure shows
m_y of both layers and the interlayer angle.
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

from micromagpy.config import MU0  # noqa: E402
from micromagpy.stack import Ellipse, Layer, Spacer, Stack  # noqa: E402

J1 = -4e-5
F, H_BIAS, H_AMP = 5e9, 100e3, 50e3


def make_stack(diameter=48e-9, cell=3e-9, interlayer_demag=True):
    disk = Ellipse(diameter, diameter)
    return Stack([Layer(disk, 1.4e-9, ms=1.0 / MU0, A=1.5e-11, K=300e3, alpha=0.011),
                  Spacer(1e-9, J1=J1),
                  Layer(disk, 3e-9, ms=1.2 / MU0, A=1.5e-11, K=800e3, alpha=0.011)], (cell, cell),
                 interlayer_demag=interlayer_demag)


def simulate(st, steps, dt):
    y = jnp.array([0.0, 1.0, 0.0])
    run = jax.jit(lambda m: st.run(m, dt, steps, h_ext=lambda t: y * (H_BIAS + H_AMP * jnp.sin(2 * jnp.pi * F * t)),
                                   solver="rk45", tol=1e-6))
    return jax.block_until_ready(run(st.uniform([(0, 0, 1), (0, 0, -1)])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=30e-9)
    args = parser.parse_args()
    st = make_stack()
    dt = 1e-12  # output interval; adaptive RK45 inside
    steps = round(args.time / dt)
    with timed("iec_coupling", mesh=[2, *st.n], steps=steps, solver="jax rk45 (stack)",
               backend=f"jax float64 on {jax.devices()[0].device_kind}") as info:
        out = simulate(st, steps, dt)
        info["rk45_steps_total"] = int(out["rk45_steps"])
    # the same without the layers' dipolar fields on each other (as a macrospin code sees it)
    m_nodip = np.asarray(simulate(make_stack(interlayer_demag=False), steps, dt)["m_avg"])
    t, m = np.asarray(out["t"]), np.asarray(out["m_avg"])  # (records, 2, 3)
    np.savez(fig_path("iec-coupling.npz"), t=t, m=m, m_no_interlayer_demag=m_nodip)
    cos = np.sum(m[:, 0] * m[:, 1], -1) / np.linalg.norm(m[:, 0], axis=-1) / np.linalg.norm(m[:, 1], axis=-1)
    angle = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    tail = t > t[-1] / 2
    print(f"layer my amplitude (last half): {m[tail, 0, 1].std():.3f} / {m[tail, 1, 1].std():.3f}, "
          f"mean angle {angle[tail].mean():.1f} deg")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(t * 1e9, m[:, 0, 1], color="crimson", lw=1, label="layer 1 $m_y$")
    a1.plot(t * 1e9, m[:, 1, 1], color="navy", lw=1, label="layer 2 $m_y$")
    a1.set_xlabel("t (ns)")
    a1.set_ylabel(r"$\langle m_y \rangle$")
    a1.legend(fontsize=8)
    a1.set_title(f"48 nm disks on 3 nm cells, J1 = {J1 * 1e3:.2f} mJ/m$^2$")
    a2.plot(t * 1e9, angle, color="forestgreen", lw=1)
    a2.axhline(180, color="r", ls="--", alpha=0.5)
    a2.set_xlabel("t (ns)")
    a2.set_ylabel("interlayer angle (deg)")
    a2.set_title(f"drive {H_BIAS / 1e3:.0f} + {H_AMP / 1e3:.0f} sin(2$\\pi$ {F / 1e9:.0f} GHz t) kA/m along y")
    fig.tight_layout()
    fig.savefig(fig_path("iec-coupling.png"), dpi=200)


if __name__ == "__main__":
    main()
