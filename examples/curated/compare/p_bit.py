"""p-bit: cmtj macrospin vs micromagpy (cf. ``switching/p_bit.py``).

12 x 12 x 2 nm PMA element, alpha = 0.5, T = 300 K, Ku set so that the
macrospin barrier is 4 kT (taken from the micromagnetic run). Both models run
the same 21 bias fields x 8 thermal replicas x 100 ns (stochastic Heun,
dt = 0.1 ps). The macrospin is a rigid moment (|m| = 1); in the 16-cell mesh
the thermal spin waves reduce the moment to m_s = <|m_z|> ~ 0.74, which
both narrows the telegraph levels and softens the sigmoid.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from cmtj import AxialDriver, SolverMode, constantDriver

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, load_micromag, macrospin, pmap, timed  # noqa: E402

n, dx = (4, 4, 1), (3e-9, 3e-9, 2e-9)
ms, damping, T, dt, time = 8e5, 0.5, 300.0, 1e-13, 100e-9


def run(args):
    H, Ku, seed = args
    j = macrospin(n, dx, ms, Ku, damping=damping)
    j.setLayerSeed("free", seed)
    j.setLayerTemperatureDriver("free", constantDriver(T))
    j.setLayerExternalFieldDriver("free", AxialDriver(constantDriver(0), constantDriver(0), constantDriver(H)))
    j.runSimulation(time, dt, 1e-12, solverMode=SolverMode.Heun)
    return np.asarray(j.getLog()["free_mz"], np.float32)


def dwell_time(mz):
    s = np.where(mz > 0.5, 1, np.where(mz < -0.5, -1, 0))
    last = np.maximum.accumulate(np.where(s != 0, np.arange(s.shape[-1]), 0), axis=-1)
    s = np.take_along_axis(s, last, axis=-1)
    return len(mz) * time / max((np.diff(s, axis=-1) != 0).sum(), 1)


def main():
    micro = load_micromag("switching", "p-bit")
    fields, Ku, h_scale = micro["fields"], float(micro["Ku"]), float(micro["h_scale"])
    replicas = 8
    jobs = [(float(H), Ku, 1000 * i + r + 1) for i, H in enumerate(fields) for r in range(replicas)]
    with timed("p_bit", trajectories=len(jobs), steps=round(time / dt), solver="cmtj Heun, process pool"):
        mz = np.array(pmap(run, jobs)).reshape(len(fields), replicas, -1)
    t = np.arange(mz.shape[-1]) * 1e-12
    settle = mz.shape[-1] // 10
    avg = mz[..., settle:].mean(axis=(1, 2))
    err = mz[..., settle:].mean(axis=2).std(axis=1) / np.sqrt(replicas)
    i0 = len(fields) // 2
    dwell = dwell_time(mz[i0])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), gridspec_kw={"width_ratios": [1.3, 1.3, 1]})
    for ax, label, tt, traces, d in [
        (axes[0], "cmtj macrospin", t, mz[i0, :3], dwell),
        (axes[1], "micromagpy, 4x4 cells", micro["t"], micro["traces"], float(micro["dwell"])),
    ]:
        for r, tr in enumerate(traces):
            ax.plot(tt * 1e9, tr + 2.4 * r, lw=0.5)
        ax.set_xlim(0, 30)
        ax.set_xlabel("t (ns)")
        ax.set_title(f"{label}: H = 0, mean dwell {d * 1e9:.2f} ns", fontsize=9)
    axes[0].set_ylabel(r"$m_z$ (offset per replica)")
    axes[1].sharey(axes[0])
    bx = axes[2]
    hh = np.linspace(fields[0], fields[-1], 200)
    m_s = float(micro["m_s"])
    bx.plot(hh / 1e3, np.tanh(hh / h_scale), "--", color="navy", lw=1, label="tanh($\\mu_0 M_s V H / kT$)")
    bx.plot(hh / 1e3, m_s * np.tanh(m_s * hh / h_scale), "--", color="crimson", lw=1, label=f"same with $m_s$ = {m_s:.2f}")
    bx.errorbar(fields / 1e3, avg, err, fmt="s", ms=4, color="navy", label="cmtj macrospin")
    bx.errorbar(fields / 1e3, micro["avg"], micro["err"], fmt="o", ms=4, color="crimson", label="micromagpy")
    bx.set_xlabel("$H_z$ (kA/m)")
    bx.set_ylabel(r"time-averaged $\langle m_z \rangle$")
    bx.set_title("p-bit sigmoid", fontsize=9)
    bx.legend(fontsize=7)
    fig.suptitle(rf"p-bit, 12x12x2 nm, $\Delta$ = 4 kT, $\alpha$ = {damping}, T = {T:g} K, {replicas} replicas x 100 ns per field", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path("p-bit.png"), dpi=200)


if __name__ == "__main__":
    main()
