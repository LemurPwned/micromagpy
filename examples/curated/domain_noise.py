"""Thermal magnetisation noise of an element made of domains with dispersed anisotropy.

The element (160 x 40 x 2 nm) is split into N bars ("domains"). Bar i has its
own uniaxial anisotropy

* magnitude ``K_i = K0 (1 + sigma xi_i)`` (xi ~ N(0, 1), clipped at 0.05 K0),
* axis tilted from the nominal direction by an angle ~ N(0, sigma) rad:
  PMA bars around z (random azimuth of the tilt), IMA bars in-plane around x,
  "mixed" alternates PMA and IMA bars.

The magnetisation is relaxed (energy minimisation) and then evolved with a
white thermal field in every cell (same global temperature, stochastic Heun).
We record the PSD of the total <m> noise, S(f) = sum_c S_<m_c>(f), and study it
versus the number of domains N and the dispersion sigma.

Usage::

    python domain_noise.py sweep     # PSD vs N and sigma for PMA and IMA (figures + npz)
    python domain_noise.py gif       # animated m_x, m_y, m_z maps (PMA, IMA, mixed)
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from _common import fig_path, timed

from micromagpy.demag import compute_demag_tensor
from micromagpy.energy import relax
from micromagpy.noise import ThermalField, power_spectral_density
from micromagpy.simulation import simulate

n = (32, 8, 1)
dx = (5e-9, 5e-9, 2e-9)
ms, A, damping = 8e5, 1.3e-11, 0.05
dt = 1e-13  # Heun; omega*dt of the fastest exchange mode ~ 0.2
TEMPERATURE = 300.0
K0 = {"pma": 8e5, "ima": 5e4}  # PMA: K_eff = K - mu0 Ms^2 / 2 ~ 4e5 J/m^3
F_DEMAG = compute_demag_tensor(n, dx)


def bar_labels(n_domains: int) -> np.ndarray:
    """Label (0..N-1) of the bar each cell belongs to; bars split the x axis."""
    ix = np.arange(n[0]) * n_domains // n[0]
    return np.broadcast_to(ix[:, None, None], n).copy()


def domain_anisotropy(mode: str, n_domains: int, sigma: float, seed: int = 0):
    """Per-cell K (n) and axis (n + (3,)), plus the per-bar values."""
    rng = np.random.default_rng(seed)
    labels = bar_labels(n_domains)
    kinds = []
    K_bar, axis_bar = np.empty(n_domains), np.empty((n_domains, 3))
    for i in range(n_domains):
        kind = mode if mode != "mixed" else ("pma" if i % 2 == 0 else "ima")
        kinds.append(kind)
        K_bar[i] = K0[kind] * max(0.05, 1 + sigma * rng.normal())
        tilt = sigma * rng.normal()
        if kind == "pma":
            phi = rng.uniform(0, 2 * np.pi)
            axis_bar[i] = [np.sin(tilt) * np.cos(phi), np.sin(tilt) * np.sin(phi), np.cos(tilt)]
        else:
            axis_bar[i] = [np.cos(tilt), np.sin(tilt), 0.0]
    return K_bar[labels], axis_bar[labels], labels, K_bar, axis_bar, kinds


def initial_state(axis_cells):
    m = axis_cells.copy()
    # pick the hemisphere: +z for PMA-like axes, +x for in-plane axes
    sign = np.where(np.abs(m[..., 2]) > 0.5, np.sign(m[..., 2]), np.sign(m[..., 0]))
    return m * sign[..., None]


def run(mode, n_domains, sigma, time, seed=0, record_every=1, callback=None):
    K, axes, labels, *_ = domain_anisotropy(mode, n_domains, sigma, seed)
    m = relax(initial_state(axes), dx, ms, A=A, K=K, kaxis=axes, demag=F_DEMAG)["m"]
    thermal = ThermalField(n, dx, ms, damping, TEMPERATURE, dt, seed=seed + 1000)
    return simulate(
        m,
        dx,
        ms,
        A,
        damping,
        dt,
        int(time / dt),
        K=K,
        kaxis=axes,
        demag=F_DEMAG,
        noise=[thermal],
        record_every=record_every,
        callback=callback,
    )


def spectrum(m_avg, discard=0.1):
    x = m_avg[int(len(m_avg) * discard) :]
    f, p = power_spectral_density(x, dt, segments=8)
    return f, p.sum(axis=1)


def metrics(f, S, f_lo=0.3e9, f_hi=60e9):
    sel = (f > f_lo) & (f < f_hi)
    w = S[sel] / S[sel].sum()
    centroid = np.sum(f[sel] * w)
    width = np.sqrt(np.sum((f[sel] - centroid) ** 2 * w))
    variance = np.trapezoid(S, f)  # total <m> noise power
    return centroid, width, variance


def smooth(S, k=5):
    return np.convolve(S, np.ones(k) / k, mode="same")


SEEDS_USED = [1]


def sweep(args):
    SEEDS_USED[0] = args.seeds
    Ns = [int(x) for x in args.domains.split(",")]
    sigmas = [float(x) for x in args.sigmas.split(",")]
    results = {}
    for mode in args.modes.split(","):
        spectra = {}
        cent = np.zeros((len(Ns), len(sigmas)))
        width = np.zeros_like(cent)
        var = np.zeros_like(cent)
        with timed(
            f"domain_noise_sweep_{mode}",
            mesh=list(n),
            runs=len(Ns) * len(sigmas) * args.seeds,
            seeds=args.seeds,
            time_per_run_ns=args.time * 1e9,
            solver="relax + heun",
            temperature_K=TEMPERATURE,
        ):
            for a, N in enumerate(Ns):
                for b, s in enumerate(sigmas):
                    # average over disorder (and thermal) realisations
                    S_all = []
                    for seed in range(args.seed, args.seed + args.seeds):
                        res = run(mode, N, s, args.time, seed=seed)
                        f, S_seed = spectrum(res["m_avg"])
                        S_all.append(S_seed)
                    S = np.mean(S_all, axis=0)
                    spectra[(N, s)] = S
                    cent[a, b], width[a, b], var[a, b] = metrics(f, S)
                    print(
                        f"{mode} N={N} sigma={s}: centroid {cent[a, b] / 1e9:.1f} GHz, "
                        f"width {width[a, b] / 1e9:.1f} GHz, var {var[a, b]:.2e}"
                    )
        results[mode] = (f, spectra, cent, width, var)
        np.savez(
            fig_path(f"domain-noise-{mode}.npz"),
            f=f,
            Ns=Ns,
            sigmas=sigmas,
            S=np.array([[spectra[(N, s)] for s in sigmas] for N in Ns]),
            centroid=cent,
            width=width,
            variance=var,
        )
        plot_mode(mode, f, spectra, Ns, sigmas, cent, width, var)


def plot_mode(mode, f, spectra, Ns, sigmas, cent, width, var):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    cmap = plt.get_cmap("viridis")
    s_fix = sigmas[-1]
    for i, N in enumerate(Ns):
        axes[0].loglog(f, smooth(spectra[(N, s_fix)]), color=cmap(i / max(1, len(Ns) - 1)), lw=1, label=f"N = {N}")
    axes[0].set_title(rf"{mode.upper()}: vs # domains ($\sigma$ = {s_fix})")
    N_fix = Ns[-1]
    for i, s in enumerate(sigmas):
        axes[1].loglog(
            f, smooth(spectra[(N_fix, s)]), color=plt.get_cmap("plasma")(i / max(1, len(sigmas) - 1)), lw=1,
            label=rf"$\sigma$ = {s}",
        )
    axes[1].set_title(rf"{mode.upper()}: vs dispersion (N = {N_fix})")
    for ax in axes[:2]:
        ax.set_xlim(5e8, 2e11)
        ax.set_xlabel("f (Hz)")
        ax.set_ylabel(r"$\sum_c S_{\langle m_c \rangle}$ (1/Hz)")
        ax.legend(fontsize=8)
    for ax, data, title, unit in (
        (axes[2], width / 1e9, "spectral width", "GHz"),
        (axes[3], var / var[0, 0], r"integrated noise $\langle\delta m^2\rangle$ / (N=1, $\sigma$=0)", ""),
    ):
        im = ax.imshow(data, origin="lower", aspect="auto", cmap="magma")
        ax.set_xticks(range(len(sigmas)), [str(s) for s in sigmas])
        ax.set_yticks(range(len(Ns)), [str(N) for N in Ns])
        ax.set_xlabel(r"$\sigma$")
        ax.set_ylabel("# domains N")
        ax.set_title(title, fontsize=9)
        for (a, b), v in np.ndenumerate(data):
            ax.text(b, a, f"{v:.1f}" if unit else f"{v:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, label=unit)
    fig.suptitle(
        f"{mode.upper()} domains, T = {TEMPERATURE:g} K, 160x40x2 nm, thermal noise only "
        f"(spectra averaged over {SEEDS_USED[0]} disorder realisations)"
    )
    fig.tight_layout()
    fig.savefig(fig_path(f"domain-noise-{mode}.png"), dpi=170)
    plt.close(fig)


def gif(args):
    from matplotlib.animation import FuncAnimation, PillowWriter

    for mode in args.modes.split(","):
        N, s = args.gif_domains, args.gif_sigma
        K, axes_c, labels, K_bar, axis_bar, kinds = domain_anisotropy(mode, N, s, args.seed)
        frames = []
        every = int(args.frame_dt / dt)

        def grab(i, t, m):
            frames.append((t, m[:, :, 0, :].copy()))

        with timed(f"domain_noise_gif_{mode}", mesh=list(n), domains=N, sigma=s, frames=int(args.gif_time / args.frame_dt)):
            run(mode, N, s, args.gif_time, seed=args.seed, record_every=every, callback=grab)
        bar_mz = np.array([[fr[1][labels[:, :, 0] == b, 2].mean() for b in range(N)] for fr in frames])
        times = np.array([fr[0] for fr in frames]) * 1e9

        fig = plt.figure(figsize=(6.4, 5.6), layout="constrained")
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.4])
        ims = []
        extent = (0, n[0] * dx[0] * 1e9, 0, n[1] * dx[1] * 1e9)
        for c, name in enumerate(("m_x", "m_y", "m_z")):
            ax = fig.add_subplot(gs[c])
            im = ax.imshow(frames[0][1][..., c].T, origin="lower", extent=extent, vmin=-1, vmax=1, cmap="RdBu_r")
            for b in range(1, N):
                ax.axvline(b * n[0] / N * dx[0] * 1e9, color="k", lw=0.8)
            for b in range(N):
                ax.text((b + 0.5) * n[0] / N * dx[0] * 1e9, extent[3] * 1.02, kinds[b].upper(), ha="center", fontsize=7)
            ax.set_ylabel(f"${name}$\ny (nm)")
            ax.set_xticks([])
            ims.append(im)
        fig.colorbar(ims[0], ax=fig.axes, shrink=0.6, location="right", label="m component")
        axt = fig.add_subplot(gs[3])
        lines = [axt.plot([], [], lw=1, label=f"bar {b}")[0] for b in range(N)]
        axt.set_xlim(0, times[-1])
        axt.set_ylim(-1.05, 1.05)
        axt.set_xlabel("t (ns)")
        axt.set_ylabel(r"bar $\langle m_z \rangle$")
        axt.legend(fontsize=6, ncol=N, loc="lower right")
        title = fig.suptitle("")

        def update(k):
            for c in range(3):
                ims[c].set_data(frames[k][1][..., c].T)
            for b, line in enumerate(lines):
                line.set_data(times[: k + 1], bar_mz[: k + 1, b])
            title.set_text(rf"{mode.upper()}, N = {N}, $\sigma$ = {s}, T = {TEMPERATURE:g} K, t = {times[k]:.2f} ns")
            return ims + lines

        anim = FuncAnimation(fig, update, frames=len(frames), blit=False)
        anim.save(fig_path(f"domain-noise-{mode}.gif"), writer=PillowWriter(fps=12), dpi=65)
        plt.close(fig)
        print("saved", fig_path(f"domain-noise-{mode}.gif"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("what", choices=["sweep", "gif"])
    parser.add_argument("--modes", default="pma,ima")
    parser.add_argument("--domains", default="1,2,4,8")
    parser.add_argument("--sigmas", default="0,0.2,0.5")
    parser.add_argument("--time", type=float, default=6e-9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=3, help="disorder realisations averaged per (N, sigma)")
    parser.add_argument("--gif-domains", type=int, default=4)
    parser.add_argument("--gif-sigma", type=float, default=0.5)
    parser.add_argument("--gif-time", type=float, default=2e-9)
    parser.add_argument("--frame-dt", type=float, default=20e-12)
    args = parser.parse_args()
    if args.what == "sweep":
        sweep(args)
    else:
        if args.modes == "pma,ima":
            args.modes = "pma,ima,mixed"
        gif(args)


if __name__ == "__main__":
    main()
