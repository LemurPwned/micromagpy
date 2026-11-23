"""cmtj vs micromagpy: Co/Pt/Co out-of-plane loops (``switching/ahe_loops.py``).

cmtj runs the curated example's LLG sweep (4 ns settle, then 15 ns per field,
dt 1e-13, alpha 0.05), with thin-film demag as written and with the squares'
own demag factors. SOT is left out on both sides (cmtj's example sets no current
driver, so its SOT terms are zero). Run the micromagnetic example first.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "switching"))
from _common import fig_path, load_micromag, macro_stack, pmap, timed  # noqa: E402
from ahe_loops import SAMPLES, ahe, make_stack, polar, sample, tilt  # noqa: E402


def cmtj_loop(args):
    k, H, matched = args
    from cmtj import AxialDriver, CVector, Junction, Layer, constantDriver

    s, J = sample(k)
    m0s = [tilt(s["K1_theta"]), tilt(s["K2_theta"])]
    if matched:
        j = macro_stack(make_stack(k), m0s)
    else:  # as in the curated example
        film = [CVector(0, 0, 0), CVector(0, 0, 0), CVector(0, 0, 1)]
        ls = [Layer(f"l{i}", CVector(*m0), CVector(*m0), s["ms_T"], 1e-9, 10e-6 * 80e-6, film, damping=0.05)
              for i, m0 in enumerate(m0s)]
        j = Junction(ls)
        j.setLayerAnisotropyDriver("l0", constantDriver(s["K1"]))
        j.setLayerAnisotropyDriver("l1", constantDriver(s["K2"]))
        j.setIECDriver("l0", "l1", constantDriver(J))
    j.runSimulation(4e-9, 1e-13, 1e-13)
    out = []
    for h in H:
        j.clearLog()
        j.setLayerExternalFieldDriver("all", AxialDriver(0, 0, float(h)))
        j.runSimulation(15e-9, 1e-13, 1e-13, calculateEnergies=False)
        log = j.getLog()
        out.append([[log[f"l{i}_m{c}"][-1] for c in "xyz"] for i in range(2)])
    return np.asarray(out)


def main():
    micro = load_micromag("switching", "ahe-loops")
    H, m_u = micro["H"], micro["m"]
    n = len(SAMPLES["t_hm"])
    jobs = [(k, H, matched) for matched in (False, True) for k in range(n)]
    with timed("ahe_loops", solver="cmtj RK4 15 ns/point", runs=len(jobs), points=len(H)):
        res = pmap(cmtj_loop, jobs)
    m_film, m_disk = np.stack(res[:n]), np.stack(res[n:])
    curves = [("cmtj, thin film (as in cmtj)", m_film, "gray", "-"),
              ("cmtj, square demag factors", m_disk, "navy", "--"),
              ("micromagpy, 64 nm squares", m_u, "crimson", "o")]
    fig, axes = plt.subplots(n, 3, figsize=(13, 9), sharex=True)
    for label, m, color, style in curves:
        kw = dict(ms=2.5, lw=0) if style == "o" else dict(lw=1.2)
        for k in range(n):
            axes[k, 0].plot(H / 1e3, ahe(m[k], k), style, color=color, label=label, **kw)
            axes[k, 1].plot(H / 1e3, polar(m[k][:, 0]), style, color=color, **kw)
            axes[k, 2].plot(H / 1e3, polar(m[k][:, 1]), style, color=color, **kw)
        diff = [np.abs(polar(m[k]) - polar(m_u[k])).max() for k in range(n)]
        print(f"{label:30s} max |theta - micromagpy| per sample: {np.round(diff, 1)} deg")
    for k in range(n):
        axes[k, 0].set_ylabel(f"Pt {SAMPLES['t_hm'][k] * 1e9:.2f} nm\nAHE " + r"$\langle m_z \rangle$")
    axes[0, 1].set_title(r"layer 1 $\theta$ (deg)")
    axes[0, 2].set_title(r"layer 2 $\theta$ (deg)")
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel(r"$H_z$ (kA/m)")
    fig.tight_layout()
    fig.savefig(fig_path("ahe-loops.png"), dpi=170)


if __name__ == "__main__":
    main()
