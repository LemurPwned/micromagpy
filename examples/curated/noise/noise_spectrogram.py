"""Noise-driven spectra vs field: thermal FMR + 1/f (micromagnetic extension).

No pulse: the dynamics of a PMA element are driven only by a white thermal
field and per-grain 1/f field noise. For each out-of-plane field the Welch PSD
of <m_x> is computed. The thermal noise excites the FMR line (which follows
the Kittel relation), while the 1/f sources show up as a low-frequency 1/f
band whose level falls as the field stiffens the magnetisation
(susceptibility ~ 1/(H + H_k,eff)^2).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, timed

from micromagpy.config import GAMMA, MU0
from micromagpy.demag import compute_demag_tensor
from micromagpy.energy import relax
from micromagpy.noise import (
    ColouredNoiseField,
    ThermalField,
    fit_spectral_exponent,
    power_spectral_density,
    voronoi_grains,
)
from micromagpy.simulation import simulate

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
    f_demag = compute_demag_tensor(n, dx)
    rng = np.random.default_rng(0)
    grains = voronoi_grains(n, 10, seed=rng)
    band = {"f_min": 1e7, "f_max": 5e11}
    psds, alphas = [], []
    m = np.zeros(n + (3,))
    m[..., 2] = 1.0
    with timed("noise_spectrogram", mesh=list(n), fields=args.fields, steps_per_field=steps, solver="heun"):
        for H in Hscan:
            m = relax(m, dx, ms, A=A, K=Ku, h_ext=(0, 0, H), demag=f_demag)["m"]
            noise = [
                ThermalField(n, dx, ms, damping, args.temperature, dt, seed=rng),
                ColouredNoiseField(n, args.onef, 1.0, dt, regions=grains, generator_kwargs=band, seed=rng),
            ]
            res = simulate(m, dx, ms, A, damping, dt, steps, K=Ku, h_ext=(0, 0, H), demag=f_demag, noise=noise)
            m = res["m"]
            f, p = power_spectral_density(res["m_avg"][steps // 10 :, 0], dt, segments=2)
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
