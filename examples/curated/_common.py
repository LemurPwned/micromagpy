"""Shared helpers for the curated examples (figure paths and timing log).

Outputs go to ``results/`` next to the running script, e.g.
``noise/noise_spectra.py`` writes to ``noise/results/``.
"""

import json
import os
import sys
import time
from contextlib import contextmanager

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(sys.modules["__main__"].__file__)), "results")
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
