"""Spectra of the 1/f^alpha noise generators (cf. cmtj ``noise_model`` plots).

For each generator we show the power spectral density with the fitted spectral
exponent, the autocorrelation and a short time trace, and record its throughput.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _common import fig_path, record_timing

from micromagpy.noise import (
    BufferedAlphaNoise,
    OneFNoise,
    OUAlphaNoise,
    VectorAlphaNoise,
    fit_spectral_exponent,
    power_spectral_density,
)

DT = 1e-12
N = 2**16
CH = 16  # independent channels, averaged in the PSD


def autocorrelation(x):
    x = x - x.mean(axis=0)
    f = np.fft.rfft(x, n=2 * len(x), axis=0)
    ac = np.fft.irfft(np.abs(f) ** 2, axis=0)[: len(x)]
    return (ac / ac[0]).mean(axis=1)


def run(gen_factory, ticks=N):
    gen = gen_factory()
    start = time.time()
    x = np.array([gen() for _ in range(ticks)])
    elapsed = time.time() - start
    return x.reshape(ticks, -1), elapsed


def main():
    generators = {
        r"Kasdin buffered, $\alpha$=1": lambda: BufferedAlphaNoise(N, 1.0, 1.0, shape=(CH,), seed=0).tick,
        r"OU sum, $\alpha$=0.5": lambda: OUAlphaNoise(0.5, 1.0, 1e6, 5e11, DT, shape=(CH,), seed=1).tick,
        r"OU sum, $\alpha$=1": lambda: OUAlphaNoise(1.0, 1.0, 1e6, 5e11, DT, shape=(CH,), seed=2).tick,
        r"OU sum, $\alpha$=1.5": lambda: OUAlphaNoise(1.5, 1.0, 1e6, 5e11, DT, shape=(CH,), seed=3).tick,
        "OneF (cmtj), 20 sources": lambda: OneFNoise(20, 0.5, 1.0, shape=(CH,), seed=4).tick,
        r"VectorAlpha x-comp, $\alpha$=1": lambda: (
            lambda v: (lambda: v.tick_vector()[..., 0])
        )(VectorAlphaNoise(N, 1.0, 1.0, normalized=False, shape=(CH,), seed=5)),
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    total = 0.0
    rates = {}
    for i, (label, factory) in enumerate(generators.items()):
        x, elapsed = run(factory)
        total += elapsed
        rates[label] = CH * N / elapsed
        f, p = power_spectral_density(x, DT, segments=8)
        p = p.mean(axis=1)
        a = fit_spectral_exponent(f, p, f_min=1e8, f_max=1e11)
        scale = 10.0 ** (-i)  # vertical offset for readability
        axes[0].loglog(f, p * scale, lw=0.8, label=f"{label}: fit {a:.2f}")
        lag = np.arange(N) * DT
        axes[1].semilogx(lag[1:], autocorrelation(x)[1:], lw=1, label=label)
        axes[2].plot(np.arange(2000) * DT * 1e9, x[:2000, 0] / x[:, 0].std() + 5 * i, lw=0.6)
        print(f"{label:35s} alpha_fit={a:.3f}  {rates[label] / 1e6:.2f} M samples/s")
    axes[0].set_xlabel("f (Hz)")
    axes[0].set_ylabel("PSD (a.u., offset)")
    axes[0].legend(fontsize=7)
    axes[0].set_title("Power spectral density")
    axes[1].set_xlabel("lag (s)")
    axes[1].set_ylabel("autocorrelation")
    axes[1].set_title("Autocorrelation")
    axes[2].set_xlabel("t (ns)")
    axes[2].set_yticks([])
    axes[2].set_title("Traces (normalised, offset)")
    fig.tight_layout()
    fig.savefig(fig_path("noise-spectra.png"), dpi=200)
    record_timing(
        "noise_spectra",
        total,
        samples=CH * N,
        generators=len(generators),
        rates_Msamples_per_s={k: round(v / 1e6, 2) for k, v in rates.items()},
    )


if __name__ == "__main__":
    main()
