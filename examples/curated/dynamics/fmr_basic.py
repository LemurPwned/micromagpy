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
    parser.add_argument("--fields", type=int, default=100)
    parser.add_argument("--hmax", type=float, default=700e3)
    parser.add_argument("--time", type=float, default=4e-9)
    args = parser.parse_args()

    Hscan = np.linspace(args.hmax, -args.hmax, args.fields)
    steps = int(args.time / dt)
    wait = int(0.2e-9 / dt)
    field = mj.make_field(n, dx, ms, A, K=Ku)
    # field slightly tilted (1 deg) to break the symmetry, pulse along x
    tilt = np.deg2rad(1.0)
    pulse = jnp.array([2e3, 0.0, 0.0])
    pulse_dur = 0.05e-9

    def point(m, H):
        h_dc = H * jnp.array([np.sin(tilt), 0.0, np.cos(tilt)])
        m = mj.relax(m, field, ms, h_ext=h_dc)["m"]
        res = mj.run(m, field, damping, dt, steps, h_ext=lambda t: h_dc + pulse * (t < pulse_dur), solver="rk45", tol=1e-5)
        return res["m"], (res["m_avg"], res["rk45_steps"])

    m0 = jnp.zeros(n + (3,)).at[..., 2].set(1.0)
    scan = jax.jit(lambda m, H: jax.lax.scan(point, m, H)[1])
    info = dict(mesh=list(n), fields=args.fields, samples_per_field=steps, solver="jax relax + rk45",
                backend=f"jax float64 on {jax.devices()[0].device_kind}")
    with timed("fmr_basic", **info) as tinfo:
        m_avg, rk_steps = jax.block_until_ready(scan(m0, jnp.asarray(Hscan)))
        tinfo["rk45_steps_total"] = int(rk_steps.sum())
    m_avg = np.asarray(m_avg)
    f = spectrum(m_avg[0, wait:, 0], dt)[0]
    spectra = np.array([spectrum(a[wait:, 0], dt)[1] for a in m_avg])
    mz_avg = m_avg[:, -1, 2]

    np.savez(fig_path("fmr-basic.npz"), H=Hscan, f=f, spectra=spectra, mz=np.array(mz_avg))
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
