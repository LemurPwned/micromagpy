"""Shared helpers for the curated examples (figure paths and timing log).

Outputs go to ``results/`` next to the running script, e.g.
``noise/noise_spectra.py`` writes to ``noise/results/``.
"""

import json
import os
import sys
import time
from contextlib import contextmanager

import numpy as np

# pool workers (spawn) have no __main__.__file__; they only compute and never write figures
_MAIN = getattr(sys.modules["__main__"], "__file__", None)
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(_MAIN or ".")), "results")
if _MAIN:
    os.makedirs(FIG_DIR, exist_ok=True)


def fig_path(name: str) -> str:
    return os.path.join(FIG_DIR, name)


def record_timing(example: str, seconds: float, **info):
    """Store the wall time of an example in results/timings.json."""
    path = fig_path("timings.json")
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[example] = {"seconds": round(seconds, 2), **info}
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    print(f"[{example}] {seconds:.1f} s {info}")


@contextmanager
def timed(example: str, **info):
    start = time.time()
    yield info
    record_timing(example, time.time() - start, **info)


def kim_resistance(m2, mxy, mz, Rx0=304.306, Ry0=1.008, AMR=-0.053, AHE=-5.71, SMR=-0.464, w_over_l=1.5):
    """Longitudinal / transverse resistance (Ohm) as cmtj ``compute_resistance`` (Kim, PRL 116, 097201).

    ``m2`` = (<m_x^2>, <m_y^2>), ``mxy`` = <m_x m_y>, ``mz`` = <m_z>, averaged over
    cells; defaults are the Ta/CoFeB Hall bar of cmtj's harmonic-Hall / spin-diode examples.
    """
    rxx = Rx0 + AMR * m2[0] + SMR * m2[1]
    rxy = Ry0 + 0.5 * AHE * mz + w_over_l * (SMR + AMR) * mxy
    return rxx, rxy


# --- macrospin (cmtj) comparisons, see compare/ ---------------------------------

CURATED = os.path.dirname(os.path.abspath(__file__))


def load_micromag(topic: str, name: str):
    """Data saved by a micromagnetic example, e.g. ``load_micromag("switching", "p-bit")``."""
    path = os.path.join(CURATED, topic, "results", f"{name}.npz")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found: run {topic}/{name.replace('-', '_')}.py first")
    return np.load(path)


def demag_factors(n, dx, mask=None):
    """(N_xx, N_yy, N_zz) of the element: cell-averaged demag field of uniform m along each axis."""
    from micromagpy.demag import compute_demag_tensor
    from micromagpy.fields import compute_hdemag

    mask = np.ones(n, bool) if mask is None else mask
    f = compute_demag_tensor(n, dx)
    return [float(-compute_hdemag(mask[..., None] * np.eye(3)[c], f, n)[mask, c].mean()) for c in range(3)]


def macrospin(n, dx, ms, K, kaxis=(0, 0, 1), damping=0.01, mask=None, m0=(0, 0, 1), stt=None):
    """cmtj junction with one layer matching a micromagnetic element.

    Same Ms (A/m), uniaxial K (J/m^3), volume and the element's demag factors
    (``demag_factors``), so the two models differ only by exchange-coupled
    non-uniform states. ``stt`` = dict(P, lam, beta) makes a Slonczewski STT layer.
    """
    from cmtj import CVector, Junction, Layer, constantDriver

    from micromagpy.config import MU0

    N = demag_factors(n, dx, mask)
    cells = n[0] * n[1] if mask is None else int(mask.sum()) // n[2]
    args = ("free", CVector(*np.asarray(m0, float) / np.linalg.norm(m0)), CVector(*kaxis), ms * MU0, n[2] * dx[2],
            cells * dx[0] * dx[1], [CVector(N[0], 0, 0), CVector(0, N[1], 0), CVector(0, 0, N[2])])
    if stt is None:
        layer = Layer(*args, damping=damping)
    else:
        layer = Layer.createSTTLayer(*args, damping=damping, SlonczewskiSpacerLayerParameter=stt["lam"],
                                     beta=stt.get("beta", 0.0), spinPolarisation=stt["P"])
    j = Junction([layer])
    j.setLayerAnisotropyDriver("free", constantDriver(K))
    return j


def pmap(fn, items):
    """Map ``fn`` over ``items`` with a process pool (fn must be a module-level function)."""
    import multiprocessing
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(mp_context=multiprocessing.get_context("spawn")) as ex:
        return list(ex.map(fn, items, chunksize=max(1, len(items) // (8 * (os.cpu_count() or 1)))))
