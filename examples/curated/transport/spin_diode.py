"""Spin-diode (VSD) field scans (cf. cmtj ``vsd-basic-field-scan.py``).

A Ta/CoFeB-like PMA element (Ms = 0.525 T, Ku = 154 kJ/m^3, alpha = 0.03,
1.45 nm thick, 128 x 128 nm on 4 nm cells) in a field tilted 1 degree from
the plane. An RF current I sin(2 pi f t) drives an out-of-plane Oersted field;
the precession it excites modulates the resistance (AMR / SMR / AHE, Kim's
formula as in cmtj) and the product I(t) R(t) has a DC part, the mixing voltage
V_mix = <I R>. Scanning the field at fixed f gives a resonance line; the
resonance fields are compared with the thin-film Kittel law
f = gamma0 / 2 pi sqrt(H (H - H_k,eff)).

Two geometries as in cmtj: "4p" reads the Hall resistance with H along x,
"2p" the longitudinal one with H at 45 degrees. All fields x frequencies x
geometries are integrated together (RK4) with the JAX backend.
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
from micromagpy.config import GAMMA, MU0  # noqa: E402

n = (32, 32, 1)
dx = (4e-9, 4e-9, 1.45e-9)
ms, Ku, A, damping = 0.525 / MU0, 1.54e5, 1.3e-11, 0.03
dt, record_every, H_OE = 1e-13, 5, 5000.0
GEOMETRIES = {"4p": (0.0, 0.75e-3), "2p": (45.0, 0.4e-3)}  # phi (deg), I_rf (A)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=int, default=60)
    parser.add_argument("--freqs", default="12,13,14,15,16", help="GHz")
    parser.add_argument("--time", type=float, default=20e-9)
    parser.add_argument("--float64", action="store_true", help="default float32 (fast on GPUs)")
    args = parser.parse_args()
    dtype = jnp.float64 if args.float64 else jnp.float32
    H = np.linspace(350e3, 630e3, args.fields)
    freqs = np.array([float(f) for f in args.freqs.split(",")]) * 1e9
    steps = round(args.time / dt)
    field = mj.make_field(n, dx, ms, A, K=Ku, kaxis=(0, 0, 1), dtype=dtype)
    m0 = jnp.zeros(n + (3,), dtype).at[..., 2].set(1.0)
    theta = np.deg2rad(91)

    def observe(m):
        return jnp.stack([(m[..., 0] ** 2).mean(), (m[..., 1] ** 2).mean(), (m[..., 0] * m[..., 1]).mean(), m[..., 2].mean()])

    def one(h, f, phi):
        hvec = h * jnp.array([jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi), jnp.cos(theta)], dtype)
        oe = jnp.array([0.0, 0.0, -H_OE], dtype)
        return mj.run(m0, field, damping, dt, steps, h_ext=lambda t: hvec + oe * jnp.sin(2 * jnp.pi * f * t),
                      solver="rk4", record_every=record_every, observe=observe)["m_avg"]

    batch = jax.jit(jax.vmap(jax.vmap(one, (0, None, None)), (None, 0, None)))  # (freq, field, time, 4)
    t = np.arange(steps // record_every) * record_every * dt
    late = t >= args.time / 2
    vmix = {}
    with timed("spin_diode", mesh=list(n), simulations=2 * len(freqs) * len(H), steps=steps,
               backend=f"jax vmap rk4 {dtype.__name__} on {jax.devices()[0].device_kind}"):
        for geo, (phi, irf) in GEOMETRIES.items():
            obs = np.asarray(batch(jnp.asarray(H, dtype), jnp.asarray(freqs, dtype), jnp.deg2rad(phi)), float)
            rxx, rxy = kim_resistance(obs[..., :2].transpose(3, 0, 1, 2), obs[..., 2], obs[..., 3])
            R = rxy if geo == "4p" else rxx
            current = irf * np.sin(2 * np.pi * freqs[:, None, None] * t)
            vmix[geo] = (current * R)[..., late].mean(-1)

    np.savez(fig_path("spin-diode.npz"), H=H, f=freqs, **vmix)
    hk_eff = 2 * Ku / (MU0 * ms) - ms
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    colors = plt.get_cmap("rainbow")(np.linspace(0, 1, len(freqs)))
    peaks = []
    for ax, geo in zip(axes, GEOMETRIES):
        for c, f, v in zip(colors, freqs, vmix[geo]):
            v = v - np.median(v)
            ax.plot(H / 1e3, v * 1e6, color=c, label=f"{f / 1e9:g} GHz")
            if geo == "2p":  # symmetric Lorentzian: its peak is the resonance field
                peaks.append(H[np.argmax(np.abs(v))])
        ax.set_xlabel("H (kA/m)")
        ax.set_ylabel(r"$V_{mix}$ ($\mu$V, offset removed)")
        ax.set_title(f"{geo}: {'Hall, H || x' if geo == '4p' else 'longitudinal, H at 45 deg'}", fontsize=9)
    axes[0].legend(fontsize=7)
    hh = np.linspace(H[0], H[-1], 200)
    axes[2].plot(hh / 1e3, GAMMA / (2 * np.pi) * np.sqrt(hh * (hh - hk_eff)) / 1e9, "k--", label="film Kittel")
    axes[2].plot(np.array(peaks) / 1e3, freqs / 1e9, "o", color="crimson", label="2p peaks")
    axes[2].set_xlabel("H (kA/m)")
    axes[2].set_ylabel("f (GHz)")
    axes[2].set_title(rf"resonance, $H_{{k,eff}}$ = {hk_eff / 1e3:.0f} kA/m (film)", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_path("spin-diode.png"), dpi=200)


if __name__ == "__main__":
    main()
