"""numba vs JAX: time per Heun step, and a batch of small thermal simulations.

``python benchmarks/benchmark_jax.py LABEL`` times numba and JAX (float64 and
float32) on whatever device JAX picks (``JAX_PLATFORMS=cpu`` forces the CPU)
and writes ``benchmarks/figures/jax_LABEL.json``. ``--plot`` combines all
``jax_*.json`` into ``benchmarks/figures/benchmark_jax.png``.
"""

import glob
import json
import multiprocessing
import os
import sys
import time

from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "benchmarks", "figures")

MESHES = [(100, 25, 1), (256, 256, 1), (64, 64, 16), (512, 512, 1), (1024, 1024, 1)]
DX, MS, A, K, ALPHA, DT = (5e-9, 5e-9, 3e-9), 8e5, 1.3e-11, 0.0, 0.02, 1e-13
BATCH, BATCH_N, BATCH_STEPS = 64, (16, 16, 1), 2000  # e.g. a 64-point field sweep of fmr_basic


def m0(n):
    m = np.random.default_rng(0).normal(size=n + (3,))
    return m / np.linalg.norm(m, axis=-1, keepdims=True)


def timeit(fn, budget=2.0, warm=True):
    if warm:
        fn()  # warm-up / compile
    reps, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < budget or reps < 2:
        fn()
        reps += 1
    return (time.perf_counter() - t0) / reps


def bench_numba():
    from micromagpy.demag import compute_demag_tensor
    from micromagpy.simulation import simulate

    rows = {}
    for n in MESHES:
        f, m, steps = compute_demag_tensor(n, DX), m0(n), 20
        t = timeit(lambda: simulate(m, DX, MS, A, ALPHA, DT, steps, demag=f, record_every=10))
        rows["x".join(map(str, n))] = t / steps * 1e3
        print("numba", n, rows["x".join(map(str, n))], "ms/step", flush=True)
    rows["batch"] = timeit(lambda: [_one_small(b) for b in range(BATCH)], budget=0, warm=False)
    print("numba batch", rows["batch"], "s", flush=True)
    # tiny meshes: thread sync dominates, so run single-threaded workers side by side
    os.environ["NUMBA_NUM_THREADS"] = os.environ["MICROMAGPY_FFT_WORKERS"] = "1"
    with ProcessPoolExecutor(os.cpu_count(), mp_context=multiprocessing.get_context("spawn")) as pool:
        list(pool.map(_one_small, range(os.cpu_count())))  # import + compile in every worker
        rows["batch_pool"] = timeit(lambda: list(pool.map(_one_small, range(BATCH))), budget=0, warm=False)
    print("numba batch, single-thread process pool", rows["batch_pool"], "s", flush=True)
    return rows


def _one_small(b):
    from micromagpy.demag import compute_demag_tensor
    from micromagpy.noise import ThermalField
    from micromagpy.simulation import simulate

    th = ThermalField(BATCH_N, DX, MS, ALPHA, 300.0, DT, seed=b)
    simulate(m0(BATCH_N), DX, MS, A, ALPHA, DT, BATCH_STEPS, K=5e5, h_ext=(0, 0, 1e3 * b),
             demag=compute_demag_tensor(BATCH_N, DX), noise=[th], record_every=10)


def bench_jax(dtype):
    import jax
    import jax.numpy as jnp

    from micromagpy import jax_backend as mj

    rows = {}
    for n in MESHES:
        field, steps = mj.make_field(n, DX, MS, A, K, dtype=dtype), 200
        run = jax.jit(lambda m: mj.run(m, field, ALPHA, DT, steps, record_every=10)["m"])
        m = jnp.asarray(m0(n), dtype)
        rows["x".join(map(str, n))] = timeit(lambda: run(m).block_until_ready()) / steps * 1e3
        print("jax", dtype.__name__, n, rows["x".join(map(str, n))], "ms/step", flush=True)
    field = mj.make_field(BATCH_N, DX, MS, A, 5e5, dtype=dtype)
    sigma = mj.thermal_sigma(DX, MS, ALPHA, 300.0, DT)

    def one(key, hz):
        return mj.run(jnp.asarray(m0(BATCH_N), dtype), field, ALPHA, DT, BATCH_STEPS,
                      h_ext=lambda t: jnp.array([0, 0, 1.0], dtype) * hz, sigma=sigma, key=key, record_every=10)["m_avg"]

    run = jax.jit(jax.vmap(one))
    keys = jax.random.split(jax.random.PRNGKey(0), BATCH)
    hz = jnp.arange(BATCH, dtype=dtype) * 1e3
    rows["batch"] = timeit(lambda: run(keys, hz).block_until_ready(), budget=0)
    print("jax", dtype.__name__, "batch", rows["batch"], "s", flush=True)
    return rows


def plot():
    import matplotlib.pyplot as plt

    data = {}
    for p in sorted(glob.glob(os.path.join(OUT, "jax_*.json"))):
        with open(p) as fh:
            data.update(json.load(fh))
    meshes = ["x".join(map(str, n)) for n in MESHES]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [3, 1.3]})
    colors = dict(zip(data, plt.get_cmap("tab10").colors))
    w = 0.8 / len(data)
    for i, (label, rows) in enumerate(data.items()):
        x = np.arange(len(meshes)) + (i - (len(data) - 1) / 2) * w
        ax.bar(x, [rows.get(m, np.nan) for m in meshes], w, label=label, color=colors[label])
    batch = [(label, rows["batch"], colors[label], "") for label, rows in data.items()]
    batch += [(label, rows["batch_pool"], colors[label], "//") for label, rows in data.items() if "batch_pool" in rows]
    for i, (label, t, c, hatch) in enumerate(batch):
        bx.barh(i, t, color=c, hatch=hatch, edgecolor="white")
        bx.text(t, i, f" {t:.3g} s", va="center", fontsize=7)
    bx.set_yticks(range(len(batch)), [lb + (" (1-thread pool)" if h else "") for lb, _, _, h in batch], fontsize=7)
    bx.invert_yaxis()
    ax.set_xticks(np.arange(len(meshes)), meshes)
    ax.set_yscale("log")
    ax.set_ylabel("ms per Heun step")
    ax.set_title("one simulation, time per step")
    ax.legend(fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    bx.set_xscale("log")
    bx.set_xlabel("s")
    bx.set_title(f"{BATCH} sims of {'x'.join(map(str, BATCH_N))}, {BATCH_STEPS} thermal steps", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "benchmark_jax.png"), dpi=200)


def main():
    if sys.argv[1] == "--plot":
        return plot()
    import jax
    import jax.numpy as jnp

    label = sys.argv[1]
    dev = jax.devices()[0].device_kind
    res = {f"{label} numba": bench_numba()} if "--no-numba" not in sys.argv else {}
    res[f"{label} JAX f64 ({dev})"] = bench_jax(jnp.float64)
    res[f"{label} JAX f32 ({dev})"] = bench_jax(jnp.float32)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, f"jax_{label}.json"), "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
