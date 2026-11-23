"""Noise-driven spectra vs field: thermal FMR + 1/f (micromagnetic extension).

No pulse: the dynamics of a PMA element are driven only by a white thermal
field and per-grain 1/f field noise. For each out-of-plane field the Welch PSD
of <m_x> is computed. The thermal noise excites the FMR line (which follows
the Kittel relation), while the 1/f sources show up as a low-frequency 1/f
band whose level falls as the field stiffens the magnetisation
(susceptibility ~ 1/(H + H_k,eff)^2). JAX backend: every field is relaxed and
integrated in one vmapped call, each with its own PRNG key.
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
from micromagpy.config import GAMMA, MU0  # noqa: E402
from micromagpy.noise import ColouredNoiseField, fit_spectral_exponent, power_spectral_density, voronoi_grains  # noqa: E402

n = (16, 16, 1)
dx = (4e-9, 4e-9, 1e-9)
ms = 1.03 / MU0
Ku = 489e3
A = 1.3e-11
damping = 0.01
dt = 5e-14  # Heun with thermal noise: keep omega*dt small for the exchange modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=int, default=11)
    parser.add_argument("--hmin", type=float, default=0.0)
    parser.add_argument("--hmax", type=float, default=500e3)
    parser.add_argument("--time", type=float, default=16e-9)
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--onef", type=float, default=2e4, help="rms 1/f field per grain (A/m)")
    args = parser.parse_args()

    Hscan = np.linspace(args.hmin, args.hmax, args.fields)
    steps = int(args.time / dt)
    field = mj.make_field(n, dx, ms, A, K=Ku)
    sigma = mj.thermal_sigma(dx, ms, damping, args.temperature, dt)
    rng = np.random.default_rng(0)
    grains = voronoi_grains(n, 10, seed=rng)
    band = {"f_min": 1e7, "f_max": 5e11}
    # one 1/f source per field (independent initial OU states); the step function is shared
    sources = [ColouredNoiseField(n, args.onef, 1.0, dt, regions=grains, generator_kwargs=band, seed=rng) for _ in Hscan]
    onef_step = mj.coloured_noise(sources[0])[1]
    onef_state = jnp.stack([mj.coloured_noise(src)[0] for src in sources])
    m0 = jnp.zeros(n + (3,)).at[..., 2].set(1.0)

    def one(H, state0, key):
        h = jnp.array([0.0, 0.0, 1.0]) * H
        m = mj.relax(m0, field, ms, h_ext=h)["m"]
        return mj.run(m, field, damping, dt, steps, h_ext=lambda t: h, sigma=sigma, noise=[(state0, onef_step)], key=key)["m_avg"]

    keys = jax.random.split(jax.random.PRNGKey(0), len(Hscan))
    with timed("noise_spectrogram", mesh=list(n), fields=args.fields, steps_per_field=steps, solver="jax heun (vmap)",
               backend=f"jax float64 on {jax.devices()[0].device_kind}"):
        m_avg = np.asarray(jax.block_until_ready(jax.jit(jax.vmap(one))(jnp.asarray(Hscan), onef_state, keys)))
    psds, alphas = [], []
    for H, a in zip(Hscan, m_avg):
        f, p = power_spectral_density(a[steps // 10 :, 0], dt, segments=2)
        psds.append(p)
        alphas.append(fit_spectral_exponent(f, p, f_max=1.5e9))
        print(f"H = {H / 1e3:6.1f} kA/m   low-f exponent {alphas[-1]:.2f}")
    psds = np.array(psds)
    np.savez(fig_path("noise-spectrogram.npz"), f=f, psd=psds, H=Hscan, alpha_low_f=alphas)
    sel = f < 40e9
    Hk_eff = 2 * Ku / (MU0 * ms) - ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))
    ax1.pcolormesh(Hscan / 1e3, f[sel] / 1e9, 10 * np.log10(psds[:, sel].T), shading="nearest", cmap="magma")
    ax1.plot(Hscan / 1e3, GAMMA / (2 * np.pi) * (Hscan + Hk_eff) / 1e9, "c--", lw=1, label="Kittel, thin-film limit")
    ax1.set_xlabel("H (kA/m)")
    ax1.set_ylabel("f (GHz)")
    ax1.set_title(f"PSD of <m_x>, T = {args.temperature:g} K + 1/f grains")
    ax1.legend(loc="upper left", fontsize=7)
    cmap = plt.get_cmap("viridis")
    for i in range(0, len(Hscan), max(1, len(Hscan) // 5)):
        ax2.loglog(
            f,
            psds[i],
            color=cmap(i / len(Hscan)),
            lw=0.8,
            label=f"H = {Hscan[i] / 1e3:.0f} kA/m",
        )
    lo = f < 1.5e9
    ax2.loglog(f[lo], psds[0][lo][0] * 2 * (f[lo] / f[lo][0]) ** -1, "k--", lw=1, label="1/f")
    ax2.set_xlabel("f (Hz)")
    ax2.set_ylabel("PSD (1/Hz)")
    ax2.legend(fontsize=7)
    ax2.set_title("Selected spectra: 1/f band below FMR, thermal FMR peak")
    fig.tight_layout()
    fig.savefig(fig_path("noise-spectrogram.png"), dpi=200)


if __name__ == "__main__":
    main()
