"""1/f noise in a perpendicular free layer (micromagnetic extension of cmtj's noise).

A PMA disk is split into Voronoi grains. Each grain carries an independent
1/f^alpha field fluctuation (e.g. fluctuating local fields / two-level
fluctuators) and an independent 1/f^alpha fluctuation of its anisotropy; a
white thermal field is added on top. We record the average magnetisation and
show that below the FMR frequency its spectrum inherits the 1/f^alpha character
of the sources (in-plane components; <mz> is second order in the fluctuations).

With the defaults (T = 0, 20 ns = 200k steps) the fitted exponent of <mx>, <my>
between 0.2 and 5 GHz is 1.00, and the FMR peak sits at ~20 GHz. At 300 K the
white thermal field of these small (4x4x2 nm) cells dominates below FMR and
hides the 1/f part unless the 1/f amplitudes are increased.

Run ``python examples/noise_sim.py --help`` for options.
"""

import argparse
import time

import numpy as np

from micromagpy.config import DTYPE
from micromagpy.demag import compute_demag_tensor
from micromagpy.noise import (
    AnisotropyNoise,
    ColouredNoiseField,
    ThermalField,
    fit_spectral_exponent,
    power_spectral_density,
    voronoi_grains,
)
from micromagpy.shapes import circular_mask, span_grid
from micromagpy.simulation import simulate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=48, help="cells along x and y")
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--dt", type=float, default=1e-13)
    parser.add_argument("--alpha-noise", type=float, default=1.0, help="spectral exponent")
    parser.add_argument("--field-noise", type=float, default=2e3, help="rms field noise per grain (A/m)")
    parser.add_argument("--dk-noise", type=float, default=2e4, help="rms anisotropy noise per grain (J/m^3)")
    parser.add_argument("--grains", type=int, default=30)
    parser.add_argument("--temperature", type=float, default=0.0, help="K; adds a white thermal field")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="noise_psd.png")
    args = parser.parse_args()

    n = (args.n, args.n, 1)
    dx = (4e-9, 4e-9, 2e-9)
    ms, A, K, damping = 1.0e6, 1.5e-11, 1.0e6, 0.05
    dt = args.dt

    # PMA disk, initially magnetised along +z
    disk = circular_mask(n[0] / 2 - 0.5, n[1] / 2 - 0.5, n[0] / 2, span_grid(n))[..., 0]
    m = np.zeros(n + (3,), dtype=DTYPE)
    m[disk, 2] = 1.0

    grains = voronoi_grains(n, args.grains, seed=args.seed) * disk
    band = {"f_min": 1.0 / (args.steps * dt) / 10, "f_max": 0.5 / dt}
    rng = np.random.default_rng(args.seed)
    noise = [
        ColouredNoiseField(
            n,
            args.field_noise,
            args.alpha_noise,
            dt,
            regions=grains,
            volume_scaling=True,
            generator_kwargs=band,
            seed=rng,
        ),
        AnisotropyNoise(
            n, ms, args.dk_noise, (0, 0, 1), args.alpha_noise, dt, regions=grains, generator_kwargs=band, seed=rng
        ),
    ]
    if args.temperature > 0:
        noise.append(ThermalField(n, dx, ms, damping, args.temperature, dt, mask=disk, seed=rng))

    start = time.time()
    f_n_demag = compute_demag_tensor(n, dx)
    res = simulate(
        m, dx, ms, A, damping, dt, args.steps, K=K, demag=f_n_demag, noise=noise, progress=True, record_every=1
    )
    print(f"Simulated {args.steps * dt * 1e9:.1f} ns in {time.time() - start:.1f} s")

    # discard the first 10% (transient), PSD of the in-plane and out-of-plane averages
    mz = res["m_avg"][args.steps // 10 :]
    f, psd = power_spectral_density(mz, dt, segments=8)
    sel_hi = 0.2e9
    fit_max = 5e9
    exps = [fit_spectral_exponent(f, psd[:, c], f_min=sel_hi, f_max=fit_max) for c in range(3)]
    print("fitted exponents (mx, my, mz) between 0.2 and 5 GHz:", np.round(exps, 2))
    np.savez(args.out.rsplit(".", 1)[0] + ".npz", f=f, psd=psd, t=res["t"], m_avg=res["m_avg"])

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5), dpi=150)
    t = res["t"] * 1e9
    for c, lab in enumerate("xyz"):
        ax1.plot(t, res["m_avg"][:, c], lw=0.6, label=f"<m{lab}>")
        ax2.loglog(f / 1e9, psd[:, c], lw=0.8, label=f"<m{lab}>, fit $\\alpha$={exps[c]:.2f}")
    ref = psd[(f > sel_hi) & (f < fit_max), 0]
    f_ref = f[(f > sel_hi) & (f < fit_max)]
    ax2.loglog(
        f_ref / 1e9,
        ref[0] * (f_ref / f_ref[0]) ** -args.alpha_noise,
        "k--",
        lw=1,
        label=f"1/f^{args.alpha_noise:g}",
    )
    ax1.set_xlabel("t (ns)")
    ax1.set_ylabel("average m")
    ax1.legend(fontsize=7)
    ax2.set_xlabel("f (GHz)")
    ax2.set_ylabel("PSD (1/Hz)")
    ax2.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
