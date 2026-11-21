"""Benchmark the original NumPy implementation against the optimised kernels.

The original tensor code is timed serially (it used a 6-process pool, 7.8 s
for 100x25x1 on a 4-core machine); larger meshes are skipped for it.

Run from the repository root: ``python benchmarks/benchmark.py``. Writes
``benchmarks/figures/benchmark.png`` and ``benchmarks/figures/benchmark.json``.
"""

import json
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from micromagpy.config import GAMMA  # noqa: E402
from micromagpy.demag import compute_demag_tensor  # noqa: E402
from micromagpy.fields import h_eff  # noqa: E402
from micromagpy.llg import pure_llg  # noqa: E402
from tests import _reference as ref  # noqa: E402

OUT = os.path.join(ROOT, "benchmarks", "figures")
os.makedirs(OUT, exist_ok=True)
MESHES = [((100, 25, 1), (5e-9, 5e-9, 3e-9)), ((256, 256, 1), (5e-9, 5e-9, 3e-9)), ((64, 64, 16), (5e-9, 5e-9, 3e-9))]


def best_of(fn, repeat=5, number=5):
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t) / number)
    return min(times)


def old_hdemag(m, f_n_demag, n, axes):
    """Original compute_hdemag (tensor already in Fourier space)."""
    m_pad = np.zeros([2 * k - 1 for k in n] + [3])
    m_pad[: n[0], : n[1], : n[2], :] = m
    f_m_pad = np.fft.fftn(m_pad, axes=axes)
    out = np.zeros(f_m_pad.shape, dtype=f_m_pad.dtype)
    out[..., 0] = (f_n_demag[..., (0, 1, 2)] * f_m_pad).sum(axis=3)
    out[..., 1] = (f_n_demag[..., (1, 3, 4)] * f_m_pad).sum(axis=3)
    out[..., 2] = (f_n_demag[..., (2, 4, 5)] * f_m_pad).sum(axis=3)
    return np.fft.ifftn(out, axes=axes)[: n[0], : n[1], : n[2], :].real


def old_step(m, f_old, n, dx):
    axes = tuple(i for i in range(3) if n[i] > 1)
    h = 8e5 * old_hdemag(m, f_old, n, axes) + ref.hex_field(1.3e-11, 8e5, m, dx, n)
    return ref.pure_llg(0.02, GAMMA, m, h, 1e-13)


def new_step(m, f, n, dx):
    h = h_eff(1.3e-11, 8e5, 0.0, (0, 0, 1), m, None, f, n, dx)
    return pure_llg(0.02, GAMMA, m, h, 1e-13)


def main():
    rng = np.random.default_rng(0)
    compute_demag_tensor((4, 3, 2), (1e-9, 1e-9, 1e-9))  # JIT warm-up
    results = []
    for n, dx in MESHES:
        m = rng.normal(size=n + (3,))
        m /= np.linalg.norm(m, axis=-1, keepdims=True)
        t = time.perf_counter()
        f_new = compute_demag_tensor(n, dx)
        t_tensor_new = time.perf_counter() - t
        if np.prod(n) <= 2500:  # the original tensor code is O(minutes) beyond this
            t = time.perf_counter()
            n_real = ref.demag_tensor_real(n, dx)
            t_tensor_old = time.perf_counter() - t
        else:  # values do not matter for the step timing
            n_real = rng.normal(size=[2 * k - 1 for k in n] + [6])
            t_tensor_old = None
        f_old = np.fft.fftn(n_real, axes=tuple(i for i in range(3) if n[i] > 1))
        t_old = best_of(lambda: old_step(m, f_old, n, dx))
        t_new = best_of(lambda: new_step(m, f_new, n, dx))
        row = {
            "mesh": "x".join(map(str, n)),
            "step_old_ms": round(t_old * 1e3, 2),
            "step_new_ms": round(t_new * 1e3, 2),
            "step_speedup": round(t_old / t_new, 1),
            "tensor_old_serial_s": None if t_tensor_old is None else round(t_tensor_old, 2),
            "tensor_new_s": round(t_tensor_new, 4),
        }
        print(row)
        results.append(row)
    with open(os.path.join(OUT, "benchmark.json"), "w") as fh:
        json.dump({"cpu_count": os.cpu_count(), "results": results}, fh, indent=2)

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(results))
    ax.bar(x - 0.2, [r["step_old_ms"] for r in results], 0.4, label="original NumPy", color="#999")
    ax.bar(x + 0.2, [r["step_new_ms"] for r in results], 0.4, label="numba + rfft", color="#1f77b4")
    for i, r in enumerate(results):
        ax.text(i + 0.2, r["step_new_ms"], f'{r["step_speedup"]}x', ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [r["mesh"] for r in results])
    ax.set_yscale("log")
    ax.set_ylabel("time per LLG step (ms)")
    ax.set_title(f"h_eff + Euler LLG step ({os.cpu_count()} cores)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "benchmark.png"), dpi=200)


if __name__ == "__main__":
    main()
