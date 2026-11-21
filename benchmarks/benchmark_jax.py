"""Benchmark the numba path against the optional JAX backend.

Per mesh (same meshes as ``benchmark.py``) it measures, per LLG step
(h_eff + update):

* numba: one Euler / Heun step called from Python;
* JAX (float64 and float32): one jitted Euler / Heun step called from Python,
  and a ``lax.scan`` of ``SCAN_STEPS`` steps compiled into one program;
* JIT compile time (``lower().compile()``) and XLA's temp-buffer estimate.

Run from the repository root: ``python benchmarks/benchmark_jax.py``.
Writes ``benchmarks/figures/benchmark_jax${BENCH_TAG}.json``; ``MESH=i``
restricts the run to one mesh, ``SCAN_STEPS`` sets the scan length (meshes
above 10k cells use a quarter of it). The committed ``benchmark_jax_all_*``
files used all 4 cores, ``benchmark_jax_1core_*`` were run under
``taskset -c 0`` with ``NUMBA_NUM_THREADS=1 MICROMAGPY_FFT_WORKERS=1``; each
was repeated until no other process used more than ~0.25 core meanwhile
(the container is shared).
"""

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from micromagpy import jax_backend as jb  # noqa: E402

jb.enable_x64()
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from micromagpy.config import GAMMA  # noqa: E402
from micromagpy.demag import compute_demag_tensor  # noqa: E402
from micromagpy.fields import h_eff  # noqa: E402
from micromagpy.llg import pure_llg, stochastic_heun  # noqa: E402

OUT = os.path.join(ROOT, "benchmarks", "figures")
MESHES = [((100, 25, 1), (5e-9, 5e-9, 3e-9)), ((256, 256, 1), (5e-9, 5e-9, 3e-9)), ((64, 64, 16), (5e-9, 5e-9, 3e-9))]
A, MS, ALPHA, DT = 1.3e-11, 8e5, 0.02, 1e-13
SCAN_STEPS = int(os.environ.get("SCAN_STEPS", "1000"))


def best_of(fn, repeat=5, number=5):
    fn()
    times = []
    for _ in range(repeat):
        t = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t) / number)
    return min(times)


def ms(x):
    return round(x * 1e3, 3)


def bench_jax(n, dx, m, f, dtype, real_tensor=False):
    model = jb.JaxModel(n, dx, MS, A, ALPHA, f_n_demag=f, dtype=dtype, real_tensor=real_tensor)
    mj = jnp.asarray(m, dtype)
    h0 = jnp.zeros(3, dtype)
    row = {}
    for solver in ("euler", "heun"):
        t = time.perf_counter()
        compiled = model.step.lower(mj, h0, DT, solver=solver).compile()
        row[f"compile_step_{solver}_s"] = round(time.perf_counter() - t, 2)
        row[f"step_{solver}_ms"] = ms(best_of(lambda: compiled(mj, h0, DT).block_until_ready()))
        mem = compiled.memory_analysis()
        if mem is not None and solver == "euler":
            row["xla_temp_MB_step"] = round(mem.temp_size_in_bytes / 2**20, 1)
        key = jax.random.PRNGKey(0)
        steps = SCAN_STEPS if np.prod(n) <= 10_000 else max(SCAN_STEPS // 4, 1)
        t = time.perf_counter()
        crun = model.run_scan.lower(mj, h0, DT, key, steps=steps, solver=solver, record_every=steps).compile()
        row[f"compile_scan_{solver}_s"] = round(time.perf_counter() - t, 2)
        row[f"scan_{solver}_ms_per_step"] = ms(best_of(lambda: crun(mj, h0, DT, key)[0].block_until_ready(), 2, 1) / steps)
        row["scan_steps"] = steps
        mem = crun.memory_analysis()
        if mem is not None and solver == "heun":
            row["xla_temp_MB_scan_heun"] = round(mem.temp_size_in_bytes / 2**20, 1)
    return row


def main():
    rng = np.random.default_rng(0)
    compute_demag_tensor((4, 3, 2), (1e-9, 1e-9, 1e-9))  # numba warm-up
    results = []
    only = os.environ.get("MESH")  # optional index into MESHES
    for idx, (n, dx) in enumerate(MESHES):
        if only is not None and int(only) != idx:
            continue
        m = rng.normal(size=n + (3,))
        m /= np.linalg.norm(m, axis=-1, keepdims=True)
        f = compute_demag_tensor(n, dx)
        hz = np.zeros(3)

        def field(mm):
            return h_eff(A, MS, 0.0, (0, 0, 1), mm, None, f, n, dx)

        row = {"mesh": "x".join(map(str, n))}
        row["numba_euler_ms"] = ms(best_of(lambda: pure_llg(ALPHA, GAMMA, m, field(m), DT)))
        row["numba_heun_ms"] = ms(best_of(lambda: stochastic_heun(ALPHA, GAMMA, m, field, hz, DT)))
        row["numba_heff_ms"] = ms(best_of(lambda: field(m)))
        jf = jb.JaxModel(n, dx, MS, A, ALPHA, f_n_demag=f).h_eff
        mj = jnp.asarray(m)
        row["jax64_heff_ms"] = ms(best_of(lambda: jf(mj).block_until_ready()))
        for tag, dtype, rt in (("jax64", jnp.float64, False), ("jax64_realN", jnp.float64, True), ("jax32", jnp.float32, False)):
            r = bench_jax(n, dx, m, f, dtype, rt)
            row.update({f"{tag}_{k}": v for k, v in r.items()})
        row["loadavg_1min"] = os.getloadavg()[0]
        print(json.dumps(row, indent=1), flush=True)
        results.append(row)
    os.makedirs(OUT, exist_ok=True)
    meta = {"cpu_count": os.cpu_count(), "jax": jax.__version__, "backend": jax.default_backend()}
    with open(os.path.join(OUT, f"benchmark_jax{os.environ.get('BENCH_TAG', '')}.json"), "w") as fh:
        json.dump({**meta, "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
