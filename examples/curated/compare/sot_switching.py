"""SOT switching loops: cmtj macrospin vs micromagpy (cf. ``switching/sot_switching.py``).

64 x 64 x 1 nm PMA element (Ku = 500 kJ/m^3, alpha = 0.1) with damping-like
SOT (p = y), in-plane bias H_x. Every point of the current sweep is a 1 ns
pulse followed by relaxation with the current off (5 ns of damped precession
for the macrospin, the energy minimiser for the mesh), read out as m_z. Both
sweeps use cmtj's ``cims-hysteresis.py`` step, 2 j_max / 130.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cmtj import AxialDriver, CVector, SolverMode, constantDriver, stepDriver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, load_micromag, macrospin, timed  # noqa: E402

n, dx = (16, 16, 1), (4e-9, 4e-9, 1e-9)
ms, K, damping, pulse = 8e5, 5e5, 0.1, 1e-9


def loop(hx, fl, sweep):
    j = macrospin(n, dx, ms, K, damping=damping)
    j.setLayerReferenceLayer("free", CVector(0, 1, 0))
    j.setLayerExternalFieldDriver("free", AxialDriver(constantDriver(hx), constantDriver(0), constantDriver(0)))
    j.runSimulation(5e-9, 1e-13, 5e-9, solverMode=SolverMode.RK4)
    mz = []
    for hdl in sweep:
        j.clearLog()
        j.setLayerDampingLikeTorqueDriver("free", stepDriver(0, hdl, 0, pulse))
        j.setLayerFieldLikeTorqueDriver("free", stepDriver(0, fl * hdl, 0, pulse))
        j.runSimulation(pulse + 5e-9, 1e-13, pulse + 5e-9, solverMode=SolverMode.RK4)
        mz.append(j.getLayerMagnetisation("free").z)
    return np.array(mz)


def main():
    micro = load_micromag("switching", "sot-switching")
    cases, hdl, jj, k = micro["cases"], micro["hdl"], micro["j"], int(micro["n_init"]) - 1
    with timed("sot_switching", loops=len(cases), points_per_loop=len(hdl), solver="cmtj RK4"):
        macro = [loop(hx, fl, hdl) for hx, fl in cases]
    for (hx, fl), a, b in zip(cases, macro, micro["mz"]):
        sw = [np.round(jj[np.where(np.diff(np.sign(m)) != 0)[0] + 1] / 1e12, 2) for m in (a, b)]
        print(f"Hx = {hx:+.0f}, Hfl/Hdl = {fl:g}: switching j (1e12 A/m^2) cmtj {sw[0]}, micromagpy {sw[1]}")

    fig, axes = plt.subplots(2, len(cases), figsize=(4.3 * len(cases), 6.5), sharex=True, sharey=True)
    for c, (hx, fl) in enumerate(cases):
        for r, (label, x, mz, k0) in enumerate([("cmtj macrospin", jj, macro[c], k),
                                                ("micromagpy 16x16", jj, micro["mz"][c], k)]):
            ax = axes[r, c]
            half = k0 + (len(x) - k0) // 2
            ax.plot(x[k0 : half + 1] / 1e12, mz[k0 : half + 1], "o-", ms=2, color="crimson", label="+j -> -j")
            ax.plot(x[half:] / 1e12, mz[half:], "s-", ms=2, color="navy", label="-j -> +j")
            ax.axhline(0, color="gray", lw=0.5)
            title = f"{label}, $H_x$ = {hx / 1e3:+.0f} kA/m"
            if fl:
                title += f", $H_{{fl}}/H_{{dl}}$ = {fl:g}"
            ax.set_title(title, fontsize=9)
        axes[1, c].set_xlabel(r"j ($10^{12}$ A/m$^2$)")
    for r in range(2):
        axes[r, 0].set_ylabel(r"$m_z$ after the pulse")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(r"SOT switching loops, 64x64x1 nm, $K_u$ = 500 kJ/m$^3$, $\alpha$ = 0.1, 1 ns pulses", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path("sot-switching.png"), dpi=200)


if __name__ == "__main__":
    main()
