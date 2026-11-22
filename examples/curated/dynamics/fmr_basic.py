"""Pulse-excited FMR field scan of a PMA element (cf. cmtj ``fmr-basic``).

cmtj's example uses three coupled macrospins; here we take the free layer
parameters (mu0 Ms = 1.03 T, Ku = 489 kJ/m^3, damping 0.01) and simulate a
micromagnetic 64 x 64 nm, 1 nm thick element with full demag and exchange.
For every out-of-plane field (scanned from +H to -H, so the reversal is
visible) a short in-plane field pulse excites the dynamics and the FFT of
<m_x> gives the spectrum. Each field point is first relaxed to equilibrium
with the energy minimiser (starting from the previous state, so the hysteresis
is kept) and the pulse response is integrated with adaptive RK45 (fixed-step
Euler is unstable here: 4 nm cells + damping 0.01). The analytic Kittel line for the uniform mode,
f = gamma0 / 2pi |H +- H_k,eff|, is overlaid.
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
from micromagpy.simulation import simulate

n = (16, 16, 1)
dx = (4e-9, 4e-9, 1e-9)
ms = 1.03 / MU0
Ku = 489e3
A = 1.3e-11
damping = 0.01
dt = 1e-12  # output interval; RK45 picks its own steps


def spectrum(mx, dt, fmax=30e9):
    mx = (mx - mx.mean()) * np.hanning(len(mx))
    y = np.abs(np.fft.rfft(mx))
    f = np.fft.rfftfreq(len(mx), dt)
    sel = (f > 0) & (f < fmax)
    return f[sel], y[sel]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=int, default=41)
    parser.add_argument("--hmax", type=float, default=700e3)
    parser.add_argument("--time", type=float, default=4e-9)
    args = parser.parse_args()

    Hscan = np.linspace(args.hmax, -args.hmax, args.fields)
    steps = int(args.time / dt)
    wait = int(0.2e-9 / dt)
    rk_steps = 0
    f_demag = compute_demag_tensor(n, dx)
    # field slightly tilted (1 deg) to break the symmetry, pulse along x
    tilt = np.deg2rad(1.0)
    pulse_amp, pulse_dur = 2e3, 0.05e-9
    m = np.zeros(n + (3,))
    m[..., 2] = 1.0
    spectra, mz_avg = [], []
    info = dict(mesh=list(n), fields=args.fields, samples_per_field=steps, solver="relax + rk45")
    with timed("fmr_basic", **info) as tinfo:
        for H in Hscan:
            h_dc = H * np.array([np.sin(tilt), 0.0, np.cos(tilt)])

            def h_ext(t, h_dc=h_dc):
                return h_dc + (np.array([pulse_amp, 0, 0]) if t < pulse_dur else 0.0)

            m = relax(m, dx, ms, A=A, K=Ku, h_ext=h_dc, demag=f_demag)["m"]
            res = simulate(
                m, dx, ms, A, damping, dt, steps, K=Ku, h_ext=h_ext, demag=f_demag, solver="rk45", tol=1e-5
            )
            rk_steps += res["rk45_steps"]
            m = res["m"]
            f, y = spectrum(res["m_avg"][wait:, 0], dt)
            spectra.append(y)
            mz_avg.append(res["m_avg"][-1, 2])
        tinfo["rk45_steps_total"] = rk_steps
    spectra = np.array(spectra)

    Hk_eff = 2 * Ku / (MU0 * ms) - ms  # thin-film estimate
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [3, 1]})
    ax.pcolormesh(Hscan / 1e3, f / 1e9, 20 * np.log10(spectra.T + 1e-12), shading="nearest", cmap="magma")
    mz = np.array(mz_avg)
    kittel = GAMMA / (2 * np.pi) * np.abs(Hscan * np.sign(mz) + Hk_eff)
    ax.plot(Hscan / 1e3, kittel / 1e9, "c--", lw=1, label="Kittel, thin-film limit")
    ax.set_ylim(0, 30)
    ax.set_xlabel("H (kA/m)")
    ax.set_ylabel("f (GHz)")
    ax.legend(loc="upper center", fontsize=7)
    ax.set_title(r"FMR, $\mu_0 M_s$=1.03 T, $K_u$=489 kJ/m$^3$, 64x64x1 nm")
    ax2.plot(Hscan / 1e3, mz, "k.-")
    ax2.set_xlabel("H (kA/m)")
    ax2.set_ylabel(r"$\langle m_z \rangle$")
    ax2.set_title("scan +H -> -H")
    fig.tight_layout()
    fig.savefig(fig_path("fmr-basic.png"), dpi=200)


if __name__ == "__main__":
    main()
