"""SOT switching phase diagram at 300 K (cf. cmtj ``cims-stability-stability-diagram.py``).

The element of ``sot_switching.py`` (64 x 64 x 1 nm PMA on a heavy metal,
damping-like SOT with p = y, theta_SH = 0.3) starts in m_z = +1. For every
pair (bias field H_x, current density j) a 1 ns current pulse is applied,
followed by 1 ns with the current off, at 300 K; the final <m_z> is averaged
over thermal replicas. The map shows the deterministic SOT switching
quadrants: the up state is switched only when j H_x has the right sign and
|j| exceeds a threshold that falls with |H_x|. Above H_dl ~ H_k,eff / 2 (j ~ 1e12 A/m^2)
the pulse pulls m in-plane towards p and the state left after the pulse is
random (overdrive). cmtj sweeps hysteresis loops
of a macrospin per field; here every (H_x, j, replica) is an independent
micromagnetic run, integrated all at once with the JAX backend (stochastic Heun).
"""

import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, timed  # noqa: E402

from micromagpy import jax_backend as mj  # noqa: E402
from micromagpy.config import MU0  # noqa: E402
from micromagpy.torques import ELECTRON_CHARGE, HBAR  # noqa: E402

n = (16, 16, 1)
dx = (4e-9, 4e-9, 1e-9)
ms, A, K, damping, T = 8e5, 1.3e-11, 5e5, 0.1, 300.0
theta_sh, dt, pulse = 0.3, 1e-13, 1e-9
HDL_PER_J = HBAR * theta_sh / (2 * ELECTRON_CHARGE * MU0 * ms * dx[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fields", type=int, default=50)
    parser.add_argument("--currents", type=int, default=50)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--hx-max", type=float, default=60e3)
    parser.add_argument("--hdl-max", type=float, default=1.5e5, help="max damping-like field (A/m); H_k,eff ~ 190 kA/m")
    parser.add_argument("--fl-ratio", type=float, default=0.0, help="Hfl / Hdl")
    parser.add_argument("--float64", action="store_true")
    args = parser.parse_args()
    dtype = jnp.float64 if args.float64 else jnp.float32
    hx = np.linspace(-args.hx_max, args.hx_max, args.fields)
    hdl = np.linspace(-args.hdl_max, args.hdl_max, args.currents)
    steps = round(2 * pulse / dt)
    field = mj.make_field(n, dx, ms, A, K=K, kaxis=(0, 0, 1), dtype=dtype)
    sigma = mj.thermal_sigma(dx, ms, damping, T, dt)
    m0 = jnp.zeros(n + (3,), dtype).at[..., 2].set(1.0)

    def one(h, d, key):
        on = lambda t: jnp.where(t < pulse, 1.0, 0.0)  # noqa: E731
        torque = mj.sot(lambda t: args.fl_ratio * d * on(t), lambda t: d * on(t), (0, 1, 0))
        out = mj.run(m0, field, damping, dt, steps, h_ext=lambda t: jnp.array([1.0, 0, 0], dtype) * h,
                     extra=torque, sigma=sigma, key=key, record_every=steps)
        return out["m"][..., 2].mean()

    keys = jax.random.split(jax.random.PRNGKey(0), args.replicas)
    batch = jax.jit(jax.vmap(jax.vmap(jax.vmap(one, (None, None, 0)), (None, 0, None)), (0, None, None)))
    with timed("sot_phase_diagram", mesh=list(n), simulations=args.fields * args.currents * args.replicas,
               steps=steps, backend=f"jax vmap heun {dtype.__name__} on {jax.devices()[0].device_kind}"):
        mz = np.asarray(batch(jnp.asarray(hx, dtype), jnp.asarray(hdl, dtype), keys), float)  # (H, j, replica)
    p_switch = (mz < 0).mean(-1)
    print(f"switched fraction over the map: {p_switch.mean():.2f}")

    j = hdl / HDL_PER_J
    np.savez(fig_path("sot-phase-diagram.npz"), hx=hx, j=j, mz=mz)
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10, 4))
    ext = [j[0] / 1e12, j[-1] / 1e12, hx[0] / 1e3, hx[-1] / 1e3]
    im = ax.imshow(mz.mean(-1), origin="lower", aspect="auto", extent=ext, cmap="RdBu", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label=r"final $\langle m_z \rangle$")
    ax.set_title(f"mean over {args.replicas} thermal replicas", fontsize=9)
    im = bx.imshow(p_switch, origin="lower", aspect="auto", extent=ext, cmap="magma", vmin=0, vmax=1)
    fig.colorbar(im, ax=bx, label="switching probability")
    bx.set_title("P(up -> down)", fontsize=9)
    for a in (ax, bx):
        a.set_xlabel(r"j ($10^{12}$ A/m$^2$)")
        a.set_ylabel("$H_x$ (kA/m)")
    fig.suptitle(f"SOT switching, 1 ns pulse, T = {T:g} K, $H_{{fl}}/H_{{dl}}$ = {args.fl_ratio:g}", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_path("sot-phase-diagram.png"), dpi=200)


if __name__ == "__main__":
    main()
