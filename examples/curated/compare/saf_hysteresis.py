"""cmtj vs micromagpy: Co/Ru/Co SAF hysteresis (``switching/saf_hysteresis.py``).

cmtj's static Smit-Beljers model (``Solver.relax_field_path``, as in
``sb-hysteresis-fast.py``) with thin-film demag, and with each layer's demag
factors from the micromagnetic disk. Run the micromagnetic example first.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "switching"))
from _common import fig_path, load_micromag, stack_demag_factors, timed  # noqa: E402
from saf_hysteresis import J1, J2, KU, MS, T1, T2, interlayer_angle, m_parallel, make_stack  # noqa: E402


def _sb_module():
    """cmtj's Smit-Beljers module; ``relax_field_path`` is newer than cmtj 1.15 on PyPI, so fall back to a
    source checkout (``CMTJ_SRC``, default ../cmtj next to this repo), loaded on top of the installed package."""
    import importlib.util

    import cmtj.models.general_sb as sb

    if hasattr(sb.Solver, "relax_field_path"):
        return sb
    root = os.environ.get("CMTJ_SRC", os.path.join(os.path.dirname(__file__), *[".."] * 4, "cmtj"))
    spec = importlib.util.spec_from_file_location("cmtj.models._general_sb_src",
                                                  os.path.join(root, "cmtj", "models", "general_sb.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sb_loops(H, angles, demag):
    """Per-layer cartesian m from the SB solver: (angles, points, layers, 3)."""
    from cmtj.models import VectorObj

    sb = _sb_module()
    LayerSB, Solver = sb.LayerSB, sb.Solver

    layers = [LayerSB(_id=i, thickness=t, Kv=VectorObj(0, 0, KU), Ks=0, Ms=MS, Ndemag=VectorObj.from_cartesian(*demag[i]))
              for i, t in enumerate((T1, T2))]
    eq = Solver(layers=layers, J1=[J1], J2=[J2]).relax_field_path(H, angles)  # (points, angles, 4)
    th, ph = eq[..., 0::2], eq[..., 1::2]
    m = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
    return np.moveaxis(m, 1, 0)


def main():
    micro = load_micromag("switching", "saf-hysteresis")
    H, angles, m_u = micro["H"], micro["angles"], micro["m"]
    N = stack_demag_factors(make_stack())
    print("layer demag factors (Nxx, Nyy, Nzz):", np.round(N, 4))
    with timed("saf_hysteresis", solver="cmtj Smit-Beljers relax_field_path", runs=2, points=len(H) * len(angles)):
        m_film = sb_loops(H, angles, [(0, 0, 1)] * 2)
        m_disk = sb_loops(H, angles, N)
    curves = [("cmtj SB, thin film (as in cmtj)", m_film, "gray", "-"),
              ("cmtj SB, disk demag factors", m_disk, "navy", "-"),
              ("micromagpy, 160 nm disk", m_u, "crimson", "o"),
              ("micromagpy, no interlayer dipolar field", micro["m_no_interlayer_demag"], "orange", "s")]
    fig, axes = plt.subplots(2, len(angles), figsize=(13, 6.5), sharex=True, sharey="row")
    for label, m, color, style in curves:
        mp, ang = m_parallel(m, angles), interlayer_angle(m)
        for k in range(len(angles)):
            kw = dict(ms=2.5, lw=0) if style in "os" else dict(lw=1.2)
            axes[0, k].plot(H / 1e3, mp[k], style, color=color, label=label, **kw)
            axes[1, k].plot(H / 1e3, ang[k], style, color=color, **kw)
        flop = [np.abs(np.diff(ang[k][: len(H) // 2])).max() for k in range(len(angles))]
        print(f"{label:40s} angle at H=0: {[round(float(a[np.argmin(np.abs(H[:len(H) // 2]))]), 1) for a in ang]}, "
              f"largest angle jump on the down sweep: {np.round(flop, 1)} deg")
    for k in range(len(angles)):
        axes[0, k].set_title(rf"$\phi_H$ = {np.degrees(angles[k]):.0f}$^\circ$")
        axes[1, k].set_xlabel("H (kA/m)")
    axes[0, 0].set_ylabel(r"$m_\parallel$ (thickness weighted)")
    axes[1, 0].set_ylabel("interlayer angle (deg)")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Co(4)/Ru(0.8)/Co(3): J1 = {J1 * 1e3:.2f}, J2 = {J2 * 1e3:.2f} mJ/m$^2$, Ku = {KU / 1e3:.0f} kJ/m$^3$ along x")
    fig.tight_layout()
    fig.savefig(fig_path("saf-hysteresis.png"), dpi=170)


if __name__ == "__main__":
    main()
