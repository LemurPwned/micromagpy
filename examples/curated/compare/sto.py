"""Spin-transfer oscillator: cmtj macrospin vs micromagpy (cf. ``dynamics/sto.py``).

In-plane free layer 64 x 64 x 2 nm, perpendicular polariser p = z, Slonczewski
STT (P = 0.5, Lambda = 1). The macrospin has the element's demag factors; both
start from the same tilted in-plane state and run 6 ns per current. The
frequency is taken from zero crossings of <m_x> over the second half.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cmtj import CVector, SolverMode, constantDriver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, load_micromag, macrospin, timed  # noqa: E402

n, dx = (16, 16, 1), (4e-9, 4e-9, 2e-9)
ms, damping, P, dt, time = 8e5, 0.02, 0.5, 1e-13, 6e-9


def run(j):
    junction = macrospin(n, dx, ms, 0.0, damping=damping, m0=(1, 0, 0.05), stt=dict(P=P, lam=1.0))
    junction.setLayerReferenceLayer("free", CVector(0, 0, 1))
    junction.setLayerCurrentDriver("free", constantDriver(j))
    junction.runSimulation(time, dt, 1e-12, solverMode=SolverMode.RK4)
    log = junction.getLog()
    return np.array(log["time"]), np.stack([log[f"free_m{c}"] for c in "xyz"], 1)


def frequency(t, m):
    x = m[len(m) // 2 :, 0]
    x = x - x.mean()
    return np.count_nonzero(np.diff(np.sign(x)) != 0) / (2 * (t[-1] - t[len(t) // 2]))


def main():
    micro = load_micromag("dynamics", "sto")
    js = np.linspace(micro["j"][0], micro["j"][-1], 25)
    with timed("sto", currents=len(js), solver="cmtj RK4"):
        runs = [run(j) for j in js]
    f_macro = np.array([frequency(t, m) for t, m in runs])
    t_ex, m_ex = run(float(micro["j_example"]))

    fig = plt.figure(figsize=(14, 4.6))
    u, v = np.mgrid[0 : np.pi : 40j, 0 : 2 * np.pi : 40j]
    for i, (label, m) in enumerate([("cmtj macrospin", m_ex), ("micromagpy <m>, 16x16 cells", micro["m_example"])]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(np.sin(u) * np.cos(v), np.sin(u) * np.sin(v), np.cos(u), alpha=0.15, color="lavender", lw=0)
        ax.plot3D(m[:, 0], m[:, 1], m[:, 2], color="crimson", lw=0.8)
        ax.scatter([m[0, 0]], [m[0, 1]], [m[0, 2]], color="forestgreen", s=30)
        ax.set_box_aspect((1, 1, 1), zoom=1.4)
        ax.view_init(elev=20)
        ax.set_axis_off()
        ax.set_title(f"{label}, j = {float(micro['j_example']):.1e} A/m$^2$", fontsize=9)
    bx = fig.add_subplot(1, 3, 3)
    bx.plot(js / 1e10, f_macro / 1e9, "-", color="navy", label="cmtj macrospin")
    bx.plot(micro["j"] / 1e10, micro["f"] / 1e9, "o", color="crimson", label="micromagpy")
    bx.plot(micro["j"] / 1e10, micro["f_theory"] / 1e9, "k--", lw=1, label=r"analytic $\gamma_0 a_J P / 4\pi\alpha$")
    bx.set_xlabel(r"j ($10^{10}$ A/m$^2$)")
    bx.set_ylabel("f (GHz)")
    bx.set_title("oscillation frequency", fontsize=9)
    bx.legend(fontsize=8)
    fig.suptitle(r"STO, 64x64x2 nm, $M_s$ = 800 kA/m, $\alpha$ = 0.02, p = z, P = 0.5", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path("sto.png"), dpi=200)


if __name__ == "__main__":
    main()
