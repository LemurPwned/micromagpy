"""Pulse-excited FMR: cmtj macrospin vs micromagpy (cf. ``dynamics/fmr_basic.py``).

64 x 64 x 1 nm PMA element (mu0 Ms = 1.03 T, Ku = 489 kJ/m^3, alpha = 0.01),
field 1 degree from the normal scanned +H -> -H. At every field the state is
relaxed, a 2 kA/m, 50 ps in-plane pulse excites it and the spectrum of m_x
over 4 ns is taken. Both scans have 100 fields, as cmtj's ``fmr-basic.py``.
The macrospin is relaxed by 5 ns of free precession instead of the energy
minimiser.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cmtj import AxialDriver, NullDriver, SolverMode, constantDriver, stepDriver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, load_micromag, macrospin, timed  # noqa: E402

from micromagpy.config import GAMMA, MU0  # noqa: E402

n, dx = (16, 16, 1), (4e-9, 4e-9, 1e-9)
ms, Ku, damping, dt, time = 1.03 / MU0, 489e3, 0.01, 1e-12, 4e-9


def spectrum(mx, fmax=30e9):  # as dynamics/fmr_basic.py
    mx = (mx - mx.mean()) * np.hanning(len(mx))
    y = np.abs(np.fft.rfft(mx))
    f = np.fft.rfftfreq(len(mx), dt)
    sel = (f > 0) & (f < fmax)
    return f[sel], y[sel]


def main():
    micro = load_micromag("dynamics", "fmr-basic")
    H = micro["H"]
    tilt = np.deg2rad(1.0)
    j = macrospin(n, dx, ms, Ku, damping=damping)
    spectra, mz = [], []
    with timed("fmr_basic", fields=len(H), solver="cmtj RK4"):
        for h in H:
            j.setLayerExternalFieldDriver("free", AxialDriver(constantDriver(h * np.sin(tilt)), constantDriver(0), constantDriver(h * np.cos(tilt))))
            j.setLayerOerstedFieldDriver("free", AxialDriver(NullDriver(), NullDriver(), NullDriver()))
            j.runSimulation(5e-9, 1e-13, 5e-9, solverMode=SolverMode.RK4)
            j.clearLog()
            j.setLayerOerstedFieldDriver("free", AxialDriver(stepDriver(0, 2e3, 0, 0.05e-9), NullDriver(), NullDriver()))
            j.runSimulation(time, 1e-13, dt, solverMode=SolverMode.RK4)
            log = j.getLog()
            f, y = spectrum(np.asarray(log["free_mx"])[200:])
            spectra.append(y)
            mz.append(log["free_mz"][-1])
    spectra, mz = np.array(spectra), np.array(mz)
    for label, h, m in (("cmtj", H, mz), ("micromagpy", micro["H"], micro["mz"])):
        i = np.argmax(m < 0)
        print(f"{label}: reversal between {h[i - 1] / 1e3:.0f} and {h[i] / 1e3:.0f} kA/m")

    hk_eff = 2 * Ku / (MU0 * ms) - ms
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [2, 2, 1]})
    for ax, label, h, ff, s, m in [
        (axes[0], "cmtj macrospin", H, f, spectra, mz),
        (axes[1], "micromagpy, 16x16 cells", micro["H"], micro["f"], micro["spectra"], micro["mz"]),
    ]:
        db = 20 * np.log10(s.T / s.max(axis=1) + 1e-12)  # each field to its own peak
        ax.pcolormesh(h / 1e3, ff / 1e9, db, shading="nearest", cmap="magma", vmin=-40, vmax=0)
        ax.plot(h / 1e3, GAMMA / (2 * np.pi) * np.abs(h * np.sign(m) + hk_eff) / 1e9, "c--", lw=0.8, label="Kittel, thin film")
        ax.set_ylim(0, 30)
        ax.set_xlabel("H (kA/m)")
        ax.set_title(label, fontsize=9)
    axes[0].set_ylabel("f (GHz)")
    axes[0].legend(loc="upper center", fontsize=7)
    axes[2].plot(H / 1e3, mz, "-", color="navy", label="cmtj")
    axes[2].plot(micro["H"] / 1e3, micro["mz"], "o", ms=3, color="crimson", label="micromagpy")
    axes[2].set_xlabel("H (kA/m)")
    axes[2].set_ylabel(r"$\langle m_z \rangle$")
    axes[2].set_title(r"scan +H $\to$ -H", fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle(r"FMR, 64x64x1 nm, $\mu_0 M_s$ = 1.03 T, $K_u$ = 489 kJ/m$^3$, $\alpha$ = 0.01 (spectra in dB of the peak at each field)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path("fmr-basic.png"), dpi=200)


if __name__ == "__main__":
    main()
