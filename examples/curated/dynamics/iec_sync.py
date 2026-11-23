"""Two PMA layers with different Ku: IEC off vs on (synchronised precession).

Two 1 nm Co-like layers (Ms = 1 MA/m, alpha = 0.005) with perpendicular
anisotropy Ku = 0.9 and 1.0 MJ/m^3 sit on a 1 nm spacer; 32 nm disks on 4 nm
cells, H_z = 100 kA/m. Both layers start tilted 10 degrees towards x and ring
down. Alone, each precesses at its own frequency, so the phase between them
runs away. With ferromagnetic IEC (J1 = +1 mJ/m^2, coupling field
J1 / (mu0 Ms t) = 0.8 MA/m, larger than the anisotropy-field difference of
0.16 MA/m) both precess at one frequency with a locked phase. The last panel
sweeps J1 (one ``vmap``) and shows each layer's dominant frequency merging.
"""

import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, timed  # noqa: E402

from micromagpy.stack import Ellipse, Layer, Spacer, Stack  # noqa: E402

KU, J_ON, HZ = (0.9e6, 1.0e6), 1e-3, 100e3
DT, STEPS = 1e-12, 5000
TILT = np.deg2rad(10)


def make_stack():
    disk = Ellipse(32e-9, 32e-9)
    return Stack([Layer(disk, 1e-9, ms=1e6, A=1.5e-11, K=KU[0], alpha=0.005), Spacer(1e-9),
                  Layer(disk, 1e-9, ms=1e6, A=1.5e-11, K=KU[1], alpha=0.005)], (4e-9, 4e-9))


def ringdown(st, J1):
    m0 = st.uniform([(np.sin(TILT), 0, np.cos(TILT))] * 2)
    h = jnp.array([0.0, 0.0, HZ])
    return st.run(m0, DT, STEPS, h_ext=lambda t: h, J1=J1, solver="rk45", tol=1e-7)["m_avg"]  # (steps, 2, 3)


def spectrum(mx):
    """|FFT| of each layer's <m_x>, (freqs,), (layers, freqs)."""
    x = mx - mx.mean(0)
    return np.fft.rfftfreq(len(x), DT), np.abs(np.fft.rfft(x * np.hanning(len(x))[:, None], axis=0)).T


def main():
    st = make_stack()
    J_sweep = np.linspace(0, 1.2e-3, 25)
    run = jax.jit(jax.vmap(lambda j: ringdown(st, j)))
    with timed("iec_sync", mesh=[2, *st.n], steps=STEPS, runs=len(J_sweep) + 2, solver="jax rk45 (stack, vmap)",
               backend=f"jax float64 on {jax.devices()[0].device_kind}"):
        m_off, m_on = np.asarray(run(jnp.array([0.0, J_ON])))
        m_sweep = np.asarray(jax.block_until_ready(run(jnp.asarray(J_sweep))))
    t = np.arange(1, STEPS + 1) * DT
    np.savez(fig_path("iec-sync.npz"), t=t, m_off=m_off, m_on=m_on, J=J_sweep, m_sweep=m_sweep)

    fig, axes = plt.subplots(2, 3, figsize=(15, 7))
    colors = ("crimson", "navy")
    for row, (label, m) in enumerate((("IEC off", m_off), ("IEC on, J1 = +1 mJ/m$^2$", m_on))):
        a = axes[row, 0]
        sel = t < 1e-9
        for k in range(2):
            a.plot(t[sel] * 1e9, m[sel, k, 0], color=colors[k], lw=1, label=f"layer {k + 1}, Ku = {KU[k] / 1e6:.1f} MJ/m$^3$")
        a.set_ylabel(r"$\langle m_x \rangle$")
        a.set_title(f"{label}: first 1 ns")
        f, S = spectrum(m[:, :, 0])
        for k in range(2):
            axes[row, 1].plot(f / 1e9, S[k] / S.max(), color=colors[k], lw=1)
            print(f"{label:28s} layer {k + 1} peak {f[np.argmax(S[k])] / 1e9:5.2f} GHz")
        axes[row, 1].set_xlim(0, 60)
        axes[row, 1].set_ylabel("|FFT $m_x$| (norm.)")
        axes[row, 1].set_title(f"{label}: spectrum (5 ns)")
        dphi = np.unwrap(np.arctan2(m[:, 0, 1], m[:, 0, 0]) - np.arctan2(m[:, 1, 1], m[:, 1, 0]))
        axes[row, 2].plot(t * 1e9, np.degrees(dphi), color="forestgreen", lw=1)
        axes[row, 2].set_ylabel(r"$\phi_1 - \phi_2$ (deg)")
        axes[row, 2].set_title(f"{label}: phase difference")
        print(f"{label:28s} phase drift over 5 ns {np.degrees(dphi[-1] - dphi[0]):9.0f} deg")
    axes[0, 0].legend(fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel("t (ns)")
    axes[1, 1].set_xlabel("f (GHz)")
    fig.tight_layout()
    fig.savefig(fig_path("iec-sync.png"), dpi=200)

    fig, ax = plt.subplots(figsize=(6, 4))
    for k in range(2):
        peaks = [spectrum(m[:, :, 0])[0][np.argmax(spectrum(m[:, :, 0])[1][k])] for m in m_sweep]
        ax.plot(J_sweep * 1e3, np.asarray(peaks) / 1e9, "o-", ms=3, color=colors[k], label=f"layer {k + 1}")
    ax.set_xlabel(r"J1 (mJ/m$^2$)")
    ax.set_ylabel("dominant frequency (GHz)")
    ax.set_title("frequency locking vs IEC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path("iec-sync-sweep.png"), dpi=200)


if __name__ == "__main__":
    main()
