"""Benchmark the original NumPy implementation against the optimised kernels.

The original tensor code is timed serially (it used a 6-process pool, 7.8 s
for 100x25x1 on a 4-core machine); larger meshes are skipped for it.

Run from the repository root: ``python benchmarks/benchmark.py <machine>``
(writes ``benchmarks/figures/benchmark_<machine>.json``), on the Mac (``mac``)
and the DGX (``dgx``), then ``python benchmarks/benchmark.py plot`` for
``benchmarks/figures/benchmark.png``.

``step_new_ms`` in the committed JSONs was measured with the former numba
kernels; since their removal, a rerun times the vectorised NumPy path instead.
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


def jax_step(n, dx, dtype):
    """The same h_eff + Euler LLG step on the JAX backend, timed inside a ``lax.scan`` of
    100 steps (as ``jax_backend.run`` integrates), so the time excludes per-call launch latency."""
    import jax
    import jax.numpy as jnp

    from micromagpy import jax_backend as mj

    field = mj.make_field(n, dx, 8e5, 1.3e-11, dtype=dtype)
    pref = GAMMA / (1 + 0.02**2)

    def step(m, _):
        c = jnp.cross(m, field(m))
        m = m - 1e-13 * pref * (c + 0.02 * jnp.cross(m, c))
        return m / jnp.linalg.norm(m, axis=-1, keepdims=True), None

    steps = jax.jit(lambda m: jax.lax.scan(step, m, None, length=100)[0])
    return lambda m: steps(m).block_until_ready()


def main(machine):
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    device = jax.devices()[0].device_kind
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
        jax_ms = {}
        for name, dtype in (("f64", jnp.float64), ("f32", jnp.float32)):
            fn, mj_ = jax_step(n, dx, dtype), jnp.asarray(m, dtype)
            jax_ms[name] = round(best_of(lambda: fn(mj_), repeat=5, number=2) / 100 * 1e3, 3)
        row = {
            "mesh": "x".join(map(str, n)),
            "step_old_ms": round(t_old * 1e3, 2),
            "step_new_ms": round(t_new * 1e3, 2),
            "step_speedup": round(t_old / t_new, 1),
            "step_jax_f64_ms": jax_ms["f64"],
            "step_jax_f32_ms": jax_ms["f32"],
            "tensor_old_serial_s": None if t_tensor_old is None else round(t_tensor_old, 2),
            "tensor_new_s": round(t_tensor_new, 4),
        }
        print(row)
        results.append(row)
    with open(os.path.join(OUT, f"benchmark_{machine}.json"), "w") as fh:
        json.dump({"machine": machine, "cpu_count": os.cpu_count(), "jax_device": device, "results": results}, fh, indent=2)


def plot():
    """Bar chart from benchmark_mac.json (NumPy, numba, JAX CPU) and benchmark_dgx.json (JAX GPU)."""
    mac = json.load(open(os.path.join(OUT, "benchmark_mac.json")))["results"]
    gpu_path = os.path.join(OUT, "benchmark_dgx.json")
    gpu = json.load(open(gpu_path))["results"] if os.path.exists(gpu_path) else None
    bars = [
        ("original NumPy (M5 Pro)", [r["step_old_ms"] for r in mac], "#999"),
        ("numba + rfft (M5 Pro, removed)", [r["step_new_ms"] for r in mac], "#1f77b4"),
        ("JAX f64 (M5 Pro CPU)", [r["step_jax_f64_ms"] for r in mac], "#2ca02c"),
    ]
    if gpu:
        bars += [("JAX f64 (GB10 GPU)", [r["step_jax_f64_ms"] for r in gpu], "#ff7f0e"),
                 ("JAX f32 (GB10 GPU)", [r["step_jax_f32_ms"] for r in gpu], "#d62728")]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x, w = np.arange(len(mac)), 0.8 / len(bars)
    for k, (label, vals, color) in enumerate(bars):
        pos = x - 0.4 + (k + 0.5) * w
        ax.bar(pos, vals, w, label=label, color=color)
        if k:
            for i, v in enumerate(vals):
                ax.text(pos[i], v, f"{mac[i]['step_old_ms'] / v:.0f}x", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [r["mesh"] for r in mac])
    ax.set_yscale("log")
    ax.set_ylabel("time per LLG step (ms)")
    ax.set_title("h_eff + Euler LLG step (speed-up over the original NumPy)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "benchmark.png"), dpi=200)


if __name__ == "__main__":
    if sys.argv[1:] == ["plot"]:
        plot()
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "mac")
