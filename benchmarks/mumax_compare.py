"""micromagpy (JAX) vs mumax3 on the same GPU: SP4, the FMR standard problem, and time per step vs mesh.

mumax3 runs from the image in ``benchmarks/mumax3/Dockerfile`` (``MUMAX_IMAGE``),
in one container started once, with the kernel cache disabled so both codes
build their demag kernels every run. Solve times subtract a run of the same
script that does all the setup and one step.

1. SP4, field 1: 128x32x1 cells, relax from (1, 0.1, 0), then 1 ns at
   mu0 H = (-24.6, 4.3, 0) mT. Adaptive Dormand-Prince, error 1e-5 per step
   in both (mumax3's default ``MaxErr``). micromagpy also runs from mumax3's
   s-state, which separates the dynamics from the relaxed state.
2. FMR standard problem (Baker et al., JMMM 421, 428 (2017)): 120x120x10 nm
   permalloy, 5 nm cells; relax in 80 kA/m along (1, 0.715, 0), then ring down
   for 20 ns in 80 kA/m along x with alpha = 0.008; spectrum of <m_y>.
3. Time per RK4 step (fixed dt = 0.1 ps, 1000 steps) for square 4 nm x 3 nm
   films from 128x128 to 2048x2048 cells; micromagpy in float64 and float32
   (mumax3 is float32).

``python benchmarks/mumax_compare.py [LABEL]`` writes
``benchmarks/figures/mumax_LABEL.{json,png}`` (LABEL defaults to the GPU name).
"""

import json
import os
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "benchmarks", "figures")
WORK = os.path.join(ROOT, "benchmarks", "mumax3", "runs")
IMAGE = os.environ.get("MUMAX_IMAGE", "lemurpwned/mumax3:3.12-cuda13")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from micromagpy import jax_backend as mj  # noqa: E402
from micromagpy.config import MU0  # noqa: E402

jax.config.update("jax_enable_x64", True)

PY = dict(ms=8e5, A=1.3e-11)
MATERIAL = "Msat = 800e3\nAex = 13e-12\n"
SP4 = dict(n=(128, 32, 1), dx=(500e-9 / 128, 125e-9 / 32, 3e-9), h1=np.array([-24.6e-3, 4.3e-3, 0.0]) / MU0)
FMR = dict(n=(24, 24, 2), dx=(5e-9,) * 3, h0=80e3 * np.array([1, 0.715, 0]) / np.hypot(1, 0.715),
           h1=np.array([80e3, 0, 0]), T=20e-9, dt=5e-12, alpha=0.008)
SCAN = dict(sizes=(128, 256, 512, 1024, 2048), dx=(4e-9, 4e-9, 3e-9), dt=1e-13, steps=1000)
TOL = 1e-5


# ---------------------------------------------------------------- mumax3 side
class Mumax:
    def __init__(self):
        os.makedirs(WORK, exist_ok=True)
        self.name = f"mumax-bench-{os.getpid()}"
        subprocess.run(["docker", "run", "-d", "--rm", "--name", self.name, "--device", "nvidia.com/gpu=all",
                        "-v", f"{WORK}:/work", "-w", "/work", IMAGE, "sleep", "infinity"], check=True, capture_output=True)

    def __call__(self, name, script):
        """Run ``script`` as ``name.mx3``; returns wall time (s)."""
        with open(os.path.join(WORK, name + ".mx3"), "w") as fh:
            fh.write(script)
        t0 = time.perf_counter()
        r = subprocess.run(["docker", "exec", self.name, "mumax3", "-cache", "", "-f", name + ".mx3"],
                           capture_output=True, text=True)
        dt = time.perf_counter() - t0
        if r.returncode:
            raise RuntimeError(r.stdout[-2000:] + r.stderr[-2000:])
        return dt

    def table(self, name):
        return np.loadtxt(os.path.join(WORK, name + ".out", "table.txt"))

    def ovf(self, name, field):
        """Load an OVF as an (nx, ny, nz, 3) array."""
        subprocess.run(["docker", "exec", self.name, "mumax3-convert", "-numpy", f"{name}.out/{field}.ovf"],
                       check=True, capture_output=True)
        a = np.load(os.path.join(WORK, name + ".out", field + ".npy"))  # (3, nz, ny, nx)
        return np.transpose(a, (3, 2, 1, 0))

    def close(self):
        subprocess.run(["docker", "rm", "-f", self.name], capture_output=True)


def mesh(n, dx):
    return f"SetGridsize({n[0]}, {n[1]}, {n[2]})\nSetCellsize({dx[0]!r}, {dx[1]!r}, {dx[2]!r})\n" + MATERIAL


def vec(h):
    return "vector({!r}, {!r}, {!r})".format(*(float(b) for b in np.asarray(h) * MU0))


def solve_time(mx, name, setup, body):
    """Wall time of ``setup + body`` minus ``setup + one step``, and the total."""
    base = mx(name + "_setup", setup + "Run(1e-15)\n")
    total = mx(name, setup + body)
    return total - base, total


# ------------------------------------------------------------ micromagpy side
def timed(fn):
    t0 = time.perf_counter()
    out = jax.block_until_ready(fn())
    return out, time.perf_counter() - t0


def py_relax(field, m0, h=0.0):
    return mj.relax(m0, field, PY["ms"], h_ext=h, torque_tol=1e-4, tol=1e-7, max_steps=10**5)["m"]


def py_run(field, m0, alpha, dt_out, steps, h):
    run = jax.jit(lambda m: mj.run(m, field, alpha, dt_out, steps, h_ext=lambda t: jnp.asarray(h), solver="rk45", tol=TOL))
    _, first = timed(lambda: run(m0))  # compile + run
    res, warm = timed(lambda: run(m0))
    return res, warm, first


# ------------------------------------------------------------------- problems
def sp4(mx):
    n, dx = SP4["n"], SP4["dx"]
    setup = mesh(n, dx) + "alpha = 0.02\nm = uniform(1, 0.1, 0)\n"
    relax_t, relax_total = solve_time(mx, "sp4_relax", setup, "Relax()\nSaveAs(m, \"s_state\")\n")
    setup_dyn = mesh(n, dx) + "alpha = 0.02\nm.LoadFile(\"sp4_relax.out/s_state.ovf\")\n" + f"B_ext = {vec(SP4['h1'])}\n"
    run_t, run_total = solve_time(mx, "sp4_run", setup_dyn, "TableAutoSave(10e-12)\nRun(1e-9)\n")
    tab = mx.table("sp4_run")
    s_mumax = mx.ovf("sp4_relax", "s_state")

    field = mj.make_field(n, dx, PY["ms"], PY["A"])
    m0 = np.zeros(n + (3,))
    m0[...] = (1, 0.1, 0)
    relax = jax.jit(lambda m: py_relax(field, m))
    _, py_relax_first = timed(lambda: relax(m0))
    s_py, py_relax_t = timed(lambda: relax(m0))
    res, py_run_t, py_run_first = py_run(field, s_py, 0.02, 10e-12, 100, SP4["h1"])
    res_mx, _, _ = py_run(field, s_mumax, 0.02, 10e-12, 100, SP4["h1"])
    return dict(
        mumax=dict(relax_s=relax_t, run_s=run_t, total_s=relax_total + run_total, t=tab[:, 0].tolist(), m=tab[:, 1:4].tolist()),
        micromagpy=dict(relax_s=py_relax_t, run_s=py_run_t, first_relax_s=py_relax_first, first_run_s=py_run_first,
                        rk45_steps=int(res["rk45_steps"]), t=np.asarray(res["t"]).tolist(), m=np.asarray(res["m_avg"]).tolist(),
                        m_from_mumax_state=np.asarray(res_mx["m_avg"]).tolist()),
        s_state_diff=float(np.abs(np.asarray(s_py) - s_mumax).max()))


def fmr(mx):
    n, dx = FMR["n"], FMR["dx"]
    steps = round(FMR["T"] / FMR["dt"])
    setup = mesh(n, dx) + f"alpha = 1\nm = uniform(1, 0.715, 0)\nB_ext = {vec(FMR['h0'])}\n"
    relax_t, _ = solve_time(mx, "fmr_relax", setup, "Relax()\nSaveAs(m, \"ground\")\n")
    setup_dyn = (mesh(n, dx) + f"alpha = {FMR['alpha']}\nm.LoadFile(\"fmr_relax.out/ground.ovf\")\n"
                 + f"B_ext = {vec(FMR['h1'])}\n")
    run_t, _ = solve_time(mx, "fmr_run", setup_dyn, f"TableAutoSave({FMR['dt']!r})\nRun({FMR['T']!r})\n")
    tab = mx.table("fmr_run")

    field = mj.make_field(n, dx, PY["ms"], PY["A"])
    m0 = np.broadcast_to(np.array([1, 0.715, 0]), n + (3,))
    relax = jax.jit(lambda m: py_relax(field, m, jnp.asarray(FMR["h0"])))
    _, _ = timed(lambda: relax(m0))
    g, py_relax_t = timed(lambda: relax(m0))
    res, py_run_t, py_run_first = py_run(field, g, FMR["alpha"], FMR["dt"], steps, FMR["h1"])
    return dict(
        mumax=dict(relax_s=relax_t, run_s=run_t, t=tab[:, 0].tolist(), m=tab[:, 1:4].tolist()),
        micromagpy=dict(relax_s=py_relax_t, run_s=py_run_t, first_run_s=py_run_first, rk45_steps=int(res["rk45_steps"]),
                        t=np.asarray(res["t"]).tolist(), m=np.asarray(res["m_avg"]).tolist()))


def scan(mx):
    rows = {"mumax3 float32": [], "micromagpy float64": [], "micromagpy float32": []}
    dx, dt, steps = SCAN["dx"], SCAN["dt"], SCAN["steps"]
    for s in SCAN["sizes"]:
        n = (s, s, 1)
        setup = mesh(n, dx) + f"alpha = 0.02\nm = uniform(1, 0.1, 0)\nSetSolver(4)\nFixDt = {dt!r}\n"
        t, _ = solve_time(mx, f"scan_{s}", setup, f"Run({steps * dt!r})\n")
        rows["mumax3 float32"].append(t / steps * 1e3)
        for dtype, key in ((jnp.float64, "micromagpy float64"), (jnp.float32, "micromagpy float32")):
            field = mj.make_field(n, dx, PY["ms"], PY["A"], dtype=dtype)
            run = jax.jit(lambda m: mj.run(m, field, 0.02, dt, steps, solver="rk4", record_every=steps)["m"])
            m0 = jnp.broadcast_to(jnp.asarray([1, 0.1, 0], dtype), n + (3,))
            timed(lambda: run(m0))
            _, t = timed(lambda: run(m0))
            rows[key].append(t / steps * 1e3)
        print("scan", s, {k: round(v[-1], 4) for k, v in rows.items()}, "ms/step", flush=True)
    return dict(sizes=list(SCAN["sizes"]), ms_per_step=rows)


def spectrum(t, my):
    """|FFT| on the uniform grid (mumax3 saves the table at the first step past each period, not on it)."""
    tu = np.arange(round(FMR["T"] / FMR["dt"])) * FMR["dt"]
    x = np.interp(tu, t, my)
    return np.fft.rfftfreq(len(x), FMR["dt"]), np.abs(np.fft.rfft(x - x.mean()))


def plot(data, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    a, d = ax[0], data["sp4"]
    for c, (lab, col) in enumerate(zip(("x", "y", "z"), ("crimson", "navy", "forestgreen"))):
        a.plot(np.asarray(d["mumax"]["t"]) * 1e9, np.asarray(d["mumax"]["m"])[:, c], color=col, lw=1.4, label=f"mumax3 <m{lab}>")
        a.plot(np.asarray(d["micromagpy"]["t"]) * 1e9, np.asarray(d["micromagpy"]["m"])[:, c], "o", ms=2.5, color=col,
               label=f"micromagpy <m{lab}>")
    a.set(xlabel="t (ns)", ylabel="<m>", title=f"SP4 field 1: run {d['mumax']['run_s']:.2f} s (mumax3) vs "
          f"{d['micromagpy']['run_s']:.2f} s (micromagpy)")
    a.legend(fontsize=7, ncol=2)
    a, d = ax[1], data["fmr"]
    for label, r, style in (("mumax3", d["mumax"], "-"), ("micromagpy", d["micromagpy"], "--")):
        f, S = spectrum(np.asarray(r["t"]), np.asarray(r["m"])[:, 1])
        a.semilogy(f / 1e9, S / S.max(), style, lw=1.2, label=f"{label} ({r['run_s']:.1f} s)")
    a.set(xlim=(0, 20), ylim=(1e-3, 2), xlabel="f (GHz)", ylabel="|FFT <m_y>| (norm.)", title="FMR standard problem, 20 ns ring-down")
    a.legend(fontsize=8)
    a, d = ax[2], data["scan"]
    cells = np.asarray(d["sizes"]) ** 2
    for (label, v), col in zip(d["ms_per_step"].items(), ("k", "crimson", "orange")):
        a.loglog(cells, v, "o-", color=col, label=label)
    a.set(xlabel="cells", ylabel="ms per RK4 step", title="time per step, square films (fixed dt)")
    a.legend(fontsize=8)
    fig.suptitle(f"micromagpy vs mumax3 on {data['device']}")
    fig.tight_layout()
    fig.savefig(path, dpi=170)


def main():
    label = sys.argv[1] if len(sys.argv) > 1 else jax.devices()[0].device_kind.replace(" ", "_")
    mx = Mumax()
    try:
        data = dict(device=jax.devices()[0].device_kind, sp4=sp4(mx), fmr=fmr(mx), scan=scan(mx))
    finally:
        mx.close()
    for p in ("sp4", "fmr"):
        print(p, {k: {q: v for q, v in d.items() if q.endswith("_s") or q == "rk45_steps"}
                  for k, d in data[p].items() if isinstance(d, dict)})
    print("SP4 max |s_state micromagpy - mumax3|:", round(data["sp4"]["s_state_diff"], 4))
    with open(os.path.join(OUT, f"mumax_{label}.json"), "w") as fh:
        json.dump(data, fh)
    plot(data, os.path.join(OUT, f"mumax_{label}.png"))


if __name__ == "__main__":
    main()
