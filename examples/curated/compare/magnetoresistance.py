"""Rxx / Rxy field sweeps: cmtj macrospin vs micromagpy (cf. ``transport/magnetoresistance.py``).

The macrospin has the element's volume, Ms, Ku and demag factors and follows
cmtj's ``harmonic-hall.py`` field sweep, relaxed for 10 ns per field with
damping 1 (cmtj uses 0.03; equilibrium states do not depend on it, and 1
settles fastest). Both use Kim's resistance formula on the (cell-averaged)
magnetisation.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cmtj import AxialDriver, constantDriver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, kim_resistance, load_micromag, macrospin, timed  # noqa: E402

from micromagpy.config import MU0  # noqa: E402

n, dx = (32, 32, 1), (4e-9, 4e-9, 1.45e-9)
ms, Ku = 0.525 / MU0, 1.54e5


def sweep(phi_deg, H):
    theta, phi = np.deg2rad(92), np.deg2rad(phi_deg)
    u = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    j = macrospin(n, dx, ms, Ku, damping=1.0)
    rows = []
    for h in H:
        j.clearLog()
        j.setLayerExternalFieldDriver("free", AxialDriver(*[constantDriver(v) for v in h * u]))
        j.runSimulation(10e-9, 1e-13, 10e-9)
        v = j.getLayerMagnetisation("free")
        m = [v.x, v.y, v.z]
        rows.append(kim_resistance((m[0] ** 2, m[1] ** 2), m[0] * m[1], m[2]))
    return np.array(rows)


def main():
    micro = load_micromag("transport", "magnetoresistance")
    H, k = micro["H"], int(micro["k"])
    with timed("magnetoresistance", points=2 * len(H), solver="cmtj RK4, 10 ns per field"):
        macro = {phi: sweep(phi, H) for phi in (1, 45)}

    hk_eff = 2 * Ku / (MU0 * ms) - ms
    fig, axes = plt.subplots(2, 4, figsize=(15, 6), sharex=True)
    for col, (phi, model, data) in enumerate(
        [(1, "cmtj", macro[1]), (1, "micromagpy", micro["phi1"]), (45, "cmtj", macro[45]), (45, "micromagpy", micro["phi45"])]
    ):
        for row, name in enumerate(("$R_{xx}$", "$R_{xy}$")):
            ax = axes[row, col]
            ax.plot(H[:k] / 1e3, data[:k, row], color="crimson", label="up")
            ax.plot(H[k:] / 1e3, data[k:, row], "--", color="navy", label="down")
            for s in (-1, 1):
                ax.axvline(s * hk_eff / 1e3, color="gray", ls=":", lw=0.8)
            ax.set_title(rf"{model}, $\phi$ = {phi}$^\circ$", fontsize=9)
            if col % 2 == 0:
                ax.set_ylabel(rf"{name} ($\Omega$)")
            else:  # same y range as the macrospin panel on its left
                ax.set_ylim(axes[row, col - 1].get_ylim())
        axes[1, col].set_xlabel("H (kA/m)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(
        rf"PMA Hall bar 128x128x1.45 nm, $\theta$ = 92$^\circ$: cmtj macrospin vs micromagpy (32x32 cells); "
        rf"dotted $\pm H_{{k,eff}}$ = {hk_eff / 1e3:.0f} kA/m",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(fig_path("magnetoresistance.png"), dpi=200)


if __name__ == "__main__":
    main()
