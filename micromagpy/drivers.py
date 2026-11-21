"""Time drivers: functions ``t -> value`` for fields, currents and torque amplitudes.

Mirrors cmtj's scalar/axial drivers. Values may be scalars, 3-vectors or
per-cell arrays. Anything accepted as an amplitude in :mod:`micromagpy`
(``simulate(h_ext=...)``, :class:`~micromagpy.torques.SOT`,
:class:`~micromagpy.torques.STT`, :class:`~micromagpy.noise.ThermalField`) is
converted once with :func:`as_driver`, so a plain number, an array, any
``callable(t)`` or one of the drivers below can be passed.
"""

from __future__ import annotations

from math import pi, sin
from typing import Any, Callable

import numpy as np

from .config import DTYPE


class Driver:
    """A value as a function of time. ``constant`` drivers allow caching."""

    def __init__(self, fn: Callable[[float], Any], constant: bool = False):
        self._fn = fn
        self.constant = constant

    def __call__(self, t: float):
        return self._fn(t)

    def __repr__(self) -> str:
        return f"Driver(constant={self.constant})"


def constant(value) -> Driver:
    """Time independent value."""
    v = np.asarray(value, dtype=DTYPE)
    return Driver(lambda t: v, constant=True)


def step(value, start: float, stop: float = np.inf, base=0.0) -> Driver:
    """``value`` for ``start <= t < stop``, ``base`` otherwise (a pulse when ``stop`` is finite)."""
    v = np.asarray(value, dtype=DTYPE)
    b = np.asarray(base, dtype=DTYPE)
    return Driver(lambda t: v if start <= t < stop else b)


def pulse(amplitude, start: float, width: float, base=0.0) -> Driver:
    """Rectangular pulse of ``width`` starting at ``start``."""
    return step(amplitude, start, start + width, base)


def sine(amplitude, frequency: float, phase: float = 0.0, offset=0.0) -> Driver:
    """``offset + amplitude * sin(2 pi f t + phase)``."""
    a = np.asarray(amplitude, dtype=DTYPE)
    o = np.asarray(offset, dtype=DTYPE)
    return Driver(lambda t: o + a * sin(2 * pi * frequency * t + phase))


def as_driver(value) -> Driver:
    """Wrap a constant, array, ``callable(t)`` or :class:`Driver` into a :class:`Driver`."""
    if isinstance(value, Driver):
        return value
    if callable(value):
        return Driver(lambda t: np.asarray(value(t), dtype=DTYPE))
    return constant(value)
