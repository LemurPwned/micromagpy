"""SOT switching phase diagram at 300 K: cmtj macrospin vs micromagpy (cf. ``switching/sot_phase_diagram.py``).

64 x 64 x 1 nm PMA element (Ku = 500 kJ/m^3, alpha = 0.1), damping-like SOT
with p = y, start in m_z = +1, 1 ns pulse + 1 ns off at 300 K (stochastic
Heun, dt = 0.1 ps), 8 thermal replicas per point. The macrospin map is on a
denser 50 x 50 grid than the micromagnetic 25 x 25 one.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cmtj import AxialDriver, CVector, SolverMode, constantDriver, stepDriver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, load_micromag, macrospin, pmap, timed  # noqa: E402

from micromagpy.config import MU0  # noqa: E402
from micromagpy.torques import ELECTRON_CHARGE, HBAR  # noqa: E402

n, dx = (16, 16, 1), (4e-9, 4e-9, 1e-9)
ms, K, damping, T, dt, pulse = 8e5, 5e5, 0.1, 300.0, 1e-13, 1e-9
HDL_PER_J = HBAR * 0.3 / (2 * ELECTRON_CHARGE * MU0 * ms * dx[2])


def run(args):
    hx, hdl, seed = args
    j = macrospin(n, dx, ms, K, damping=damping)
    j.setLayerSeed("free", seed)
    j.setLayerTemperatureDriver("free", constantDriver(T))
    j.setLayerReferenceLayer("free", CVector(0, 1, 0))
    j.setLayerDampingLikeTorqueDriver("free", stepDriver(0, hdl, 0, pulse))
    j.setLayerExternalFieldDriver("free", AxialDriver(constantDriver(hx), constantDriver(0), constantDriver(0)))
    j.runSimulation(2 * pulse, dt, 2 * pulse, solverMode=SolverMode.Heun)
    return j.getLayerMagnetisation("free").z


def main():
    micro = load_micromag("switching", "sot-phase-diagram")
    hx = np.linspace(micro["hx"][0], micro["hx"][-1], 50)
    j_scan = np.linspace(micro["j"][0], micro["j"][-1], 50)
    replicas = micro["mz"].shape[-1]
    jobs = [(float(h), float(j * HDL_PER_J), 1 + i) for i, (h, j, _) in
            enumerate((h, j, r) for h in hx for j in j_scan for r in range(replicas))]
    with timed("sot_phase_diagram", simulations=len(jobs), steps=round(2 * pulse / dt), solver="cmtj Heun, process pool"):
        mz = np.array(pmap(run, jobs)).reshape(len(hx), len(j_scan), replicas)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, label, h, j, m in [
        (axes[0], f"cmtj macrospin, {len(hx)}x{len(j_scan)}", hx, j_scan, mz),
        (axes[1], f"micromagpy 16x16 cells, {len(micro['hx'])}x{len(micro['j'])}", micro["hx"], micro["j"], micro["mz"]),
    ]:
        ext = [j[0] / 1e12, j[-1] / 1e12, h[0] / 1e3, h[-1] / 1e3]
        im = ax.imshow((m < 0).mean(-1), origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=1)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(r"j ($10^{12}$ A/m$^2$)")
    axes[0].set_ylabel("$H_x$ (kA/m)")
    fig.colorbar(im, ax=axes, label="P(up -> down)")
    fig.suptitle(f"SOT switching probability, 1 ns pulse, T = {T:g} K, {replicas} replicas", fontsize=10)
    fig.savefig(fig_path("sot-phase-diagram.png"), dpi=200, bbox_inches="tight")


if __name__ == "__main__":
    main()
