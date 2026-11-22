"""Probabilistic bit: thermally switching low-barrier PMA nanodisk (cf. cmtj ``p-bit.py``).

A 12 x 12 x 2 nm PMA element (4 x 4 cells) at 300 K with an energy barrier of a few kT
flips randomly between m_z = +1 and -1 (telegraph noise). A bias field H_z
tilts the double well, and the time-averaged <m_z> follows the two-state
Boltzmann law <m_z> = m_s tanh(mu0 Ms m_s V H / kT), with m_s = <|m_z|> the
thermally reduced moment of the element: the sigmoid of a p-bit. cmtj's
example biases a macrospin with STT and field; here the bias is the field
and every cell is resolved (demag, exchange, thermal field per cell).

All fields x replicas run as one batch with the JAX backend
(:mod:`micromagpy.jax_backend`, ``jax.vmap``), which is where a GPU pays off:
each trajectory is tiny, but there are hundreds of them. ``--numba-estimate``
times a short numba run of one trajectory for comparison.
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, timed  # noqa: E402

from micromagpy import jax_backend as mj  # noqa: E402
from micromagpy.config import BOLTZMANN, MU0  # noqa: E402
from micromagpy.demag import compute_demag_tensor  # noqa: E402
from micromagpy.fields import compute_hdemag  # noqa: E402

n = (4, 4, 1)
dx = (3e-9, 3e-9, 2e-9)
ms, A, damping, T = 8e5, 1.3e-11, 0.5, 300.0  # large damping: fast attempts, dwell >> relaxation
dt, record_every = 1e-13, 10


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, default=4.0, help="target barrier K_eff V / kT")
    parser.add_argument("--time", type=float, default=100e-9)
    parser.add_argument("--fields", type=int, default=21)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--numba-estimate", action="store_true")
    args = parser.parse_args()

    disk = np.ones(n, bool)
    V = disk.sum() * np.prod(dx)
    kT = BOLTZMANN * T
    # shape anisotropy of the disk from the demag field of uniform m_z and m_x
    f_demag = compute_demag_tensor(n, dx)
    nz_minus_nx = [-compute_hdemag(disk[..., None] * np.eye(3)[c], f_demag, n)[disk, c].mean() for c in (2, 0)]
    nz_minus_nx = nz_minus_nx[0] - nz_minus_nx[1]
    Ku = args.delta * kT / V + 0.5 * MU0 * ms**2 * nz_minus_nx
    h_scale = kT / (MU0 * ms * V)  # field at which the Zeeman energy is kT
    fields = np.linspace(-3, 3, args.fields) * h_scale
    steps = round(args.time / dt)

    m0 = np.zeros(n + (3,))
    m0[disk] = (0, 0, 1)
    field = mj.make_field(n, dx, ms, A, K=Ku, kaxis=(0, 0, 1))
    sigma = mj.thermal_sigma(dx, ms, damping, T, dt)

    def one(H, key):
        return mj.run(m0, field, damping, dt, steps, h_ext=lambda t: jnp.array([0.0, 0.0, 1.0]) * H,
                      sigma=sigma, key=key, record_every=record_every)["m_avg"][:, 2]

    batch = jax.jit(jax.vmap(jax.vmap(one, (None, 0)), (0, None)))
    keys = jax.random.split(jax.random.PRNGKey(1), args.replicas)
    with timed("p_bit", mesh=list(n), trajectories=args.fields * args.replicas, steps=steps,
               backend=f"jax vmap on {jax.devices()[0].device_kind}") as info:
        mz = np.asarray(batch(jnp.asarray(fields), keys))  # (fields, replicas, time)
        if args.numba_estimate:
            from micromagpy.noise import ThermalField
            from micromagpy.simulation import simulate

            th = ThermalField(n, dx, ms, damping, T, dt, mask=disk)
            t0, probe = time.time(), 2000
            simulate(m0, dx, ms, A, damping, dt, probe, K=Ku, demag=f_demag, noise=[th], record_every=record_every)
            info["numba_estimate_s"] = round((time.time() - t0) / probe * steps * args.fields * args.replicas, 1)

    m_s = np.abs(mz[args.fields // 2]).mean()
    t = np.arange(mz.shape[-1]) * record_every * dt
    settle = len(t) // 10
    avg = mz[..., settle:].mean(axis=(1, 2))
    err = mz[..., settle:].mean(axis=2).std(axis=1) / np.sqrt(args.replicas)
    i0 = args.fields // 2
    # binary state with +-0.5 hysteresis thresholds, so noise around m_z = 0 is not counted
    s = np.where(mz[i0] > 0.5, 1, np.where(mz[i0] < -0.5, -1, 0))
    last = np.maximum.accumulate(np.where(s != 0, np.arange(s.shape[-1]), 0), axis=-1)
    s = np.take_along_axis(s, last, axis=-1)
    flips = (np.diff(s, axis=-1) != 0).sum()
    dwell = args.replicas * args.time / max(flips, 1)
    print(f"Ku = {Ku / 1e3:.0f} kJ/m^3, V = {V * 1e27:.0f} nm^3, H(kT) = {h_scale:.0f} A/m, "
          f"mean dwell time at H = 0: {dwell * 1e9:.2f} ns, m_s = {m_s:.2f}")

    np.savez(fig_path("p-bit.npz"), fields=fields, avg=avg, err=err, t=t, traces=mz[i0, :3],
             m_s=m_s, h_scale=h_scale, dwell=dwell, Ku=Ku)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1.6, 1]})
    for r in range(min(3, args.replicas)):
        ax.plot(t * 1e9, mz[i0, r] + 2.4 * r, lw=0.6)
    ax.set_xlabel("t (ns)")
    ax.set_ylabel(r"$\langle m_z \rangle$ (offset per replica)")
    ax.set_title(f"telegraph noise at H = 0, $\\Delta$ = {args.delta:g} kT, mean dwell {dwell * 1e9:.2f} ns", fontsize=9)
    hh = np.linspace(fields[0], fields[-1], 200)
    bx.plot(hh / 1e3, m_s * np.tanh(m_s * hh / h_scale), "k--", lw=1,
            label=rf"$m_s$ tanh($\mu_0 M_s m_s V H / kT$), $m_s$ = {m_s:.2f}")
    bx.errorbar(fields / 1e3, avg, err, fmt="o", ms=4, color="crimson", label=f"{args.replicas} replicas x {args.time * 1e9:g} ns")
    bx.set_xlabel("$H_z$ (kA/m)")
    bx.set_ylabel(r"time-averaged $\langle m_z \rangle$")
    bx.set_title(f"p-bit sigmoid, T = {T:g} K", fontsize=9)
    bx.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_path("p-bit.png"), dpi=200)


if __name__ == "__main__":
    main()
