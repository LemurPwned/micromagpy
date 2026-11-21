"""Shared helpers for the curated examples (figure paths and timing log)."""

import json
import os
import time
from contextlib import contextmanager

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(FIG_DIR, exist_ok=True)


def fig_path(name: str) -> str:
    return os.path.join(FIG_DIR, name)


def record_timing(example: str, seconds: float, **info):
    """Store the wall time of an example in figures/timings.json."""
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
