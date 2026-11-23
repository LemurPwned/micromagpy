"""cmtj vs micromagpy: IEC-coupled oscillations (``dynamics/iec_coupling.py``).

cmtj runs two macrospins: the curated example as written (thin-film demag), and
one matched to the micromagnetic disks (each layer's own demag factors, same
volume; cmtj has no dipolar field between layers). Run the micromagnetic
example first.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dynamics"))
from _common import fig_path, load_micromag, macro_stack, stack_demag_factors, timed  # noqa: E402
from iec_coupling import F, H_AMP, H_BIAS, J1, make_stack  # noqa: E402


def drive(j, ids):
    from cmtj import AxialDriver, ScalarDriver

    for lid in ids:
        j.setLayerExternalFieldDriver(lid, AxialDriver(ScalarDriver.getConstantDriver(0),
                                                       ScalarDriver.getSineDriver(H_BIAS, H_AMP, F, 0),
                                                       ScalarDriver.getConstantDriver(0)))


def run_cmtj(j, ids, T):
    drive(j, ids)
    j.runSimulation(T, 1e-12, 1e-12)
    log = j.getLog()
    t = np.asarray(log["time"])
    m = np.stack([np.stack([np.asarray(log[f"{i}_m{c}"]) for c in "xyz"], -1) for i in ids], 1)
    return t, m


def thin_film(T):
    """The curated example as written."""
    from cmtj import CVector, Junction, Layer, constantDriver

    demag = [CVector(0, 0, 0), CVector(0, 0, 0), CVector(0, 0, 1.0)]
    l1 = Layer("layer1", CVector(0, 0, 1), CVector(0, 0, 1), 1.0, 1.4e-9, 7e-10 * 7e-10, demag, damping=0.011)
    l2 = Layer("layer2", CVector(0, 0, -1), CVector(0, 0, 1), 1.2, 3e-9, 7e-10 * 7e-10, demag, damping=0.011)
    l1.setAnisotropyDriver(constantDriver(300e3))
    l2.setAnisotropyDriver(constantDriver(800e3))
    j = Junction([l1, l2])
    j.setIECDriver("layer1", "layer2", constantDriver(J1))
    return run_cmtj(j, ["layer1", "layer2"], T)


def angle(m):
    cos = np.sum(m[:, 0] * m[:, 1], -1) / np.linalg.norm(m[:, 0], axis=-1) / np.linalg.norm(m[:, 1], axis=-1)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    micro = load_micromag("dynamics", "iec-coupling")
    st = make_stack()
    T = float(micro["t"][-1] + micro["t"][1])
    with timed("iec_coupling", solver="cmtj RK4", runs=2, time_ns=T * 1e9):
        t_m, m_m = run_cmtj(macro_stack(st, [(0, 0, 1), (0, 0, -1)]), ["l0", "l1"], T)
        t_f, m_f = thin_film(T)
    N = stack_demag_factors(st)
    print("layer demag factors (Nxx, Nyy, Nzz):", np.round(N, 3))
    t_u, m_u = micro["t"], micro["m"]
    curves = [("cmtj, thin film (as in cmtj)", t_f, m_f, "gray", 0.8),
              ("cmtj, matched to the disks", t_m, m_m, "navy", 1.0),
              ("micromagpy, 48 nm disks", t_u, m_u, "crimson", 1.0),
              ("micromagpy, no interlayer dipolar field", t_u, micro["m_no_interlayer_demag"], "orange", 0.8)]
    for label, t, m, *_ in curves:
        tail = t > t[-1] / 2
        print(f"{label:40s} my amplitude {m[tail, 0, 1].std():.3f} / {m[tail, 1, 1].std():.3f}, "
              f"mean angle {angle(m)[tail].mean():6.1f} deg, final angle {angle(m)[-1]:6.1f}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 6.5))
    for row, (t0, t1) in enumerate([(0, T), (T - 2e-9, T)]):
        for label, t, m, color, lw in curves:
            sel = (t >= t0) & (t <= t1)
            axes[row, 0].plot(t[sel] * 1e9, m[sel, 0, 1], color=color, lw=lw, label=label)
            axes[row, 1].plot(t[sel] * 1e9, m[sel, 1, 1], color=color, lw=lw)
            axes[row, 2].plot(t[sel] * 1e9, angle(m)[sel], color=color, lw=lw)
        axes[row, 0].set_ylabel(r"layer 1 $m_y$")
        axes[row, 1].set_ylabel(r"layer 2 $m_y$")
        axes[row, 2].set_ylabel("interlayer angle (deg)")
    for ax in axes[1]:
        ax.set_xlabel("t (ns)")
    axes[0, 0].legend(fontsize=7)
    axes[0, 1].set_title("full run")
    axes[1, 1].set_title("last 2 ns")
    fig.suptitle(f"IEC-coupled oscillations, J1 = {J1 * 1e3:.2f} mJ/m$^2$, drive along y "
                 f"{H_BIAS / 1e3:.0f} + {H_AMP / 1e3:.0f} sin(2$\\pi$ {F / 1e9:.0f} GHz t) kA/m")
    fig.tight_layout()
    fig.savefig(fig_path("iec-coupling.png"), dpi=170)


if __name__ == "__main__":
    main()
