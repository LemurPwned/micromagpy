"""Hysteresis of an antiferromagnetically coupled Co/Ru/Co bilayer (cf. cmtj ``sb-hysteresis-fast.py``).

cmtj computes the loops with its static Smit-Beljers model (two macrospins,
thin-film demag). Here the bilayer is a 160 nm disk (in-plane isotropic shape,
so a film-like loop) on 4 nm cells: Co layers of 4 and 3 nm (Ms = 1.65 T, weak
uniaxial Ku = 2 kJ/m^3 along x), coupled through 0.8 nm of Ru with
J1 = -0.12 mJ/m^2 (bilinear) and J2 = -0.05 mJ/m^2 (biquadratic). The in-plane
field is swept +80 -> -80 -> +80 kA/m at 0, 30 and 60 degrees from the easy
axis; every point is relaxed from the previous state (one ``lax.scan`` per
angle, the angles in one ``vmap``). The figure shows the net moment along the
field (thickness weighted) and the angle between the layers.
"""

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

MS, KU, A = 1.65 / MU0, 2e3, 1.5e-11
T1, T2, J1, J2 = 4e-9, 3e-9, -1.2e-4, -5e-5
HMAX, N_HALF = 80e3, 40
ANGLES_DEG = np.array([0.0, 30.0, 60.0])


def make_stack(diameter=160e-9, cell=4e-9, interlayer_demag=True):
    disk = Ellipse(diameter, diameter)
    return Stack([Layer(disk, T1, ms=MS, A=A, K=KU, kaxis=(1, 0, 0)), Spacer(0.8e-9, J1=J1, J2=J2),
                  Layer(disk, T2, ms=MS, A=A, K=KU, kaxis=(1, 0, 0))], (cell, cell), interlayer_demag=interlayer_demag)


def field_path():
    down = np.linspace(HMAX, -HMAX, N_HALF)
    return np.concatenate([down, down[::-1]])


def loops(st, H, angles):
    """Relaxed per-layer <m> along the sweep, for every field angle: (angles, points, layers, 3)."""

    def one(phi):
        u = jnp.array([jnp.cos(phi), jnp.sin(phi), 0.0])
        # along the first field, the layers split by +-0.01 rad to break the symmetry
        d = phi + jnp.array([1e-2, -1e-2])
        m0 = jnp.stack([jnp.cos(d), jnp.sin(d), jnp.zeros(2)], -1)[:, None, None, :] * st.masks[..., None]

        def point(m, h):
            r = st.relax(m, h_ext=h * u, torque_tol=1e-4, tol=1e-6)
            return r["m"], (st.layer_average(r["m"]), r["steps"])

        return jax.lax.scan(point, m0, H)[1]

    return jax.jit(jax.vmap(one))(jnp.asarray(angles))


def m_parallel(m_avg, angles):
    """Net moment along the field, thickness weighted: (angles, points)."""
    u = np.stack([np.cos(angles), np.sin(angles), np.zeros_like(angles)], -1)[:, None, None, :]
    return np.average(np.sum(m_avg * u, -1), axis=-1, weights=[T1, T2])


def interlayer_angle(m_avg):
    a, b = m_avg[..., 0, :], m_avg[..., 1, :]
    cos = np.sum(a * b, -1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    st = make_stack()
    H, angles = field_path(), np.deg2rad(ANGLES_DEG)
    with timed("saf_hysteresis", mesh=[2, *st.n], points=len(H) * len(angles), solver="jax relax (stack, vmap)",
               backend=f"jax float64 on {jax.devices()[0].device_kind}") as info:
        m_avg, steps = jax.block_until_ready(loops(st, jnp.asarray(H), angles))
        info["relax_steps_total"] = int(steps.sum())
    m_avg = np.asarray(m_avg)
    # the same without the layers' dipolar fields on each other (as a macrospin / SB model sees it)
    m_nodip = np.asarray(loops(make_stack(interlayer_demag=False), jnp.asarray(H), angles)[0])
    np.savez(fig_path("saf-hysteresis.npz"), H=H, angles=angles, m=m_avg, m_no_interlayer_demag=m_nodip)
    mpar, ang = m_parallel(m_avg, angles), interlayer_angle(m_avg)

    fig, axes = plt.subplots(2, len(angles), figsize=(12, 6), sharex=True, sharey="row")
    for k, deg in enumerate(ANGLES_DEG):
        axes[0, k].plot(H / 1e3, mpar[k], "-o", ms=2, lw=1, color="crimson")
        axes[1, k].plot(H / 1e3, ang[k], "-o", ms=2, lw=1, color="navy")
        axes[0, k].set_title(rf"$\phi_H$ = {deg:.0f}$^\circ$")
        axes[1, k].set_xlabel("H (kA/m)")
    axes[0, 0].set_ylabel(r"$m_\parallel$ (thickness weighted)")
    axes[1, 0].set_ylabel("interlayer angle (deg)")
    fig.suptitle(f"Co(4)/Ru(0.8)/Co(3) disk, 160 nm, J1 = {J1 * 1e3:.2f}, J2 = {J2 * 1e3:.2f} mJ/m$^2$")
    fig.tight_layout()
    fig.savefig(fig_path("saf-hysteresis.png"), dpi=200)
    for k, deg in enumerate(ANGLES_DEG):
        print(f"phi_H = {deg:4.0f}: angle at H=0 (descending) {ang[k][np.argmin(np.abs(H[:N_HALF]))]:.1f} deg, "
              f"m_par at +Hmax {mpar[k][0]:.3f}")


if __name__ == "__main__":
    main()
