"""Out-of-plane loops of Co/Pt/Co with IEC across the Pt (cf. cmtj ``ahe_loops.py``).

Four samples (Pt 0.70 / 1.88 / 2.59 / 3.48 nm) with two 1 nm Co layers, different
Ms, perpendicular Ku of each layer (axes tilted by under 1 degree) and IEC through
the Pt, J = J0 * t_Co / (t_Pt + t_Co). Hz is swept -7000 -> +7000 -> -7000 Oe
(30 points each way), each point relaxed from the previous state. The layers are
64 nm squares on 4 nm cells. cmtj's example also creates SOT layers, but it never
sets a current driver, so its torques are zero; they are left out here. The
figure shows the AHE signal (AHE-weighted <m_z>) and both layers' polar angles.
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, timed  # noqa: E402

from micromagpy.config import MU0  # noqa: E402
from micromagpy.stack import Bar, Layer, Spacer, Stack  # noqa: E402

OE = 1e3 / (4 * np.pi)  # A/m per Oe
T_FM = 1e-9
SAMPLES = dict(t_hm=np.array([0.7, 1.88, 2.59, 3.48]) * 1e-9, ms_T=[1.1, 1.18, 1.05, 1.15],
               K1=np.array([0.55, 1.0, 0.625, 0.625]) * 1e6, K2=np.array([0.3, 0.27, 0.29, 0.27]) * 1e6,
               K1_theta=[1, 0.01, 0.1, 0.1], K2_theta=[0.01, 0.01, 0.5, 0.5],
               J0=np.array([8, 6, 3.5, 0.3]) * 1e-3, ahe_asym=[1.0, 1.0, 0.8, 0.5])
H_SCAN = np.linspace(-7000, 7000, 30) * OE


def tilt(theta_deg):
    th = np.deg2rad(theta_deg)
    return (float(np.sin(th)), 0.0, float(np.cos(th)))


def sample(k):
    s = {key: v[k] for key, v in SAMPLES.items()}
    J = s["J0"] * T_FM / (s["t_hm"] + T_FM)
    return s, J


def make_stack(k, side=64e-9, cell=4e-9):
    s, J = sample(k)
    sq = Bar(side, side)
    ms = s["ms_T"] / MU0
    return Stack([Layer(sq, T_FM, ms=ms, A=1.5e-11, K=s["K1"], kaxis=tilt(s["K1_theta"]), alpha=0.05),
                  Spacer(s["t_hm"], J1=J),
                  Layer(sq, T_FM, ms=ms, A=1.5e-11, K=s["K2"], kaxis=tilt(s["K2_theta"]), alpha=0.05)], (cell, cell))


def field_path():
    return np.concatenate([H_SCAN, H_SCAN[::-1]])


def loop(st, m0, H):
    z = jnp.array([0.0, 0.0, 1.0])

    def point(m, h):
        r = st.relax(m, h_ext=h * z, torque_tol=1e-4, tol=1e-6)
        return r["m"], st.layer_average(r["m"])

    return jax.jit(lambda m: jax.lax.scan(point, m, jnp.asarray(H))[1])(st.relax(m0, tol=1e-6)["m"])


def ahe(m_avg, k):
    return np.average(m_avg[..., 2], axis=-1, weights=[1.0, SAMPLES["ahe_asym"][k]])


def polar(m_avg):
    return np.degrees(np.arccos(np.clip(m_avg[..., 2] / np.linalg.norm(m_avg, axis=-1), -1, 1)))


def main():
    H = field_path()
    out = []
    for k in range(len(SAMPLES["t_hm"])):
        st = make_stack(k)
        s, _ = sample(k)
        with timed(f"ahe_loops sample {k}", mesh=[2, *st.n], points=len(H), solver="jax relax (stack)",
                   backend=f"jax float64 on {jax.devices()[0].device_kind}"):
            m0 = st.uniform([tilt(s["K1_theta"]), tilt(s["K2_theta"])])
            out.append(np.asarray(jax.block_until_ready(loop(st, m0, H))))
    m = np.stack(out)  # (samples, points, layers, 3)
    np.savez(fig_path("ahe-loops.npz"), H=H, m=m)

    fig, axes = plt.subplots(len(out), 2, figsize=(9, 8), sharex=True)
    for k in range(len(out)):
        axes[k, 0].plot(H / 1e3, ahe(m[k], k), "-o", ms=2, lw=1, color="k")
        axes[k, 0].set_ylabel(r"AHE $\langle m_z \rangle$")
        axes[k, 0].set_title(f"Pt {SAMPLES['t_hm'][k] * 1e9:.2f} nm", fontsize=9)
        axes[k, 1].plot(H / 1e3, polar(m[k][:, 0]), "o", ms=2, color="forestgreen", label="top")
        axes[k, 1].plot(H / 1e3, polar(m[k][:, 1]), "o", ms=2, color="crimson", label="bottom")
        axes[k, 1].set_ylabel(r"$\theta$ (deg)")
    axes[0, 1].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel(r"$H_z$ (kA/m)")
    fig.tight_layout()
    fig.savefig(fig_path("ahe-loops.png"), dpi=200)


if __name__ == "__main__":
    main()
