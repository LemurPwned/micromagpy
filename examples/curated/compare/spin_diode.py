"""Spin-diode field scans: cmtj macrospin vs micromagpy (cf. ``transport/spin_diode.py``).

128 x 128 x 1.45 nm PMA element (Ms = 0.525 T, Ku = 154 kJ/m^3, alpha = 0.03)
in a field 1 degree from the plane, driven by an RF Oersted field of 5 kA/m.
Both use the same 20 ns RK4 runs and V_mix = <I R> over the second half with
Kim's resistance formula. The macrospin scan has 100 fields (as cmtj's
``vsd-basic-field-scan.py``), the micromagnetic one 60.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cmtj import AxialDriver, NullDriver, SolverMode, constantDriver, sineDriver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, kim_resistance, load_micromag, macrospin, pmap, timed  # noqa: E402

from micromagpy.config import GAMMA, MU0  # noqa: E402

n, dx = (32, 32, 1), (4e-9, 4e-9, 1.45e-9)
ms, Ku, damping, dt, time, H_OE = 0.525 / MU0, 1.54e5, 0.03, 1e-13, 20e-9, 5000.0
GEOMETRIES = {"4p": (0.0, 0.75e-3), "2p": (45.0, 0.4e-3)}  # phi (deg), I_rf (A)


def run(args):
    h, f, geo = args
    phi, irf = GEOMETRIES[geo]
    theta, phi = np.deg2rad(91), np.deg2rad(phi)
    u = [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    j = macrospin(n, dx, ms, Ku, damping=damping)
    j.setLayerExternalFieldDriver("free", AxialDriver(*[constantDriver(h * c) for c in u]))
    j.setLayerOerstedFieldDriver("free", AxialDriver(NullDriver(), NullDriver(), sineDriver(0, -H_OE, f, 0)))
    j.runSimulation(time, dt, 5 * dt, solverMode=SolverMode.RK4)
    log = j.getLog()
    t = np.asarray(log["time"])
    m = [np.asarray(log[f"free_m{c}"]) for c in "xyz"]
    rxx, rxy = kim_resistance((m[0] ** 2, m[1] ** 2), m[0] * m[1], m[2])
    late = t >= time / 2
    current = irf * np.sin(2 * np.pi * f * t[late])
    return float(np.mean(current * (rxy if geo == "4p" else rxx)[late]))


def main():
    micro = load_micromag("transport", "spin-diode")
    freqs = micro["f"]
    H = np.linspace(micro["H"][0], micro["H"][-1], 100)
    jobs = [(float(h), float(f), g) for g in GEOMETRIES for f in freqs for h in H]
    with timed("spin_diode", simulations=len(jobs), steps=round(time / dt), solver="cmtj RK4, process pool"):
        v = np.array(pmap(run, jobs)).reshape(len(GEOMETRIES), len(freqs), len(H))
    macro = dict(zip(GEOMETRIES, v))

    hk_eff = 2 * Ku / (MU0 * ms) - ms
    colors = plt.get_cmap("rainbow")(np.linspace(0, 1, len(freqs)))
    fig, axes = plt.subplots(1, 5, figsize=(22, 4), gridspec_kw={"width_ratios": [1, 1, 1, 1, 0.9]})
    peaks = {}
    for k, (model, data, hs) in enumerate([("cmtj", macro, H), ("micromagpy", micro, micro["H"])]):
        peaks[model] = []
        for g, geo in enumerate(GEOMETRIES):
            ax = axes[2 * g + k]
            for c, f, vv in zip(colors, freqs, data[geo]):
                vv = vv - np.median(vv)
                ax.plot(hs / 1e3, vv * 1e6, color=c, label=f"{f / 1e9:g} GHz")
                if geo == "2p":
                    peaks[model].append(hs[np.argmax(np.abs(vv))])
            ax.set_title(f"{model}, {geo}", fontsize=9)
            ax.set_xlabel("H (kA/m)")
    for g in range(2):  # same V range for the two models of one geometry
        lo = min(axes[2 * g].get_ylim()[0], axes[2 * g + 1].get_ylim()[0])
        hi = max(axes[2 * g].get_ylim()[1], axes[2 * g + 1].get_ylim()[1])
        for ax in axes[2 * g : 2 * g + 2]:
            ax.set_ylim(lo, hi)
    axes[0].set_ylabel(r"$V_{mix}$ ($\mu$V, offset removed)")
    axes[0].legend(fontsize=7)
    hh = np.linspace(H[0], H[-1], 200)
    axes[4].plot(hh / 1e3, GAMMA / (2 * np.pi) * np.sqrt(hh * (hh - hk_eff)) / 1e9, "k--", lw=1, label="film Kittel")
    axes[4].plot(np.array(peaks["cmtj"]) / 1e3, freqs / 1e9, "s", color="navy", label="cmtj 2p peaks")
    axes[4].plot(np.array(peaks["micromagpy"]) / 1e3, freqs / 1e9, "o", color="crimson", label="micromagpy 2p peaks")
    axes[4].set_xlabel("H (kA/m)")
    axes[4].set_ylabel("f (GHz)")
    axes[4].set_title("resonance", fontsize=9)
    axes[4].legend(fontsize=7)
    fig.suptitle("Spin diode, 128x128x1.45 nm: cmtj macrospin vs micromagpy (32x32 cells)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path("spin-diode.png"), dpi=200)


if __name__ == "__main__":
    main()
