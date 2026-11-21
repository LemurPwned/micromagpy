"""Coloured (1/f^alpha) and thermal noise for micromagnetic simulations.

Temporal generators
-------------------
These mirror the generators of `cmtj <https://github.com/LemurPwned/cmtj>`_
(``core/noise.hpp``) and add a streaming generator better suited to meshes:

* :func:`alpha_noise` / :class:`BufferedAlphaNoise` -- Kasdin's fractional
  integration filter (Kasdin 1995, Proc. IEEE 83, 802). White Gaussian noise is
  convolved with the filter ``h_0 = 1, h_k = h_{k-1} (alpha/2 + k - 1) / k``
  (FFT convolution), which gives a PSD ~ 1/f^alpha over the whole buffer.
  Same algorithm as cmtj's ``BufferedAlphaNoise``. Each buffer is generated
  independently, so nothing slower than ``1/(buffer_size dt)`` is produced.
* :class:`VectorAlphaNoise` -- three buffered components, optionally normalised
  (cmtj's ``VectorAlphaNoise``).
* :class:`OneFNoise` -- cmtj's ``OneFNoise``: a set of ``sources`` uniform
  random states; on each tick a geometric number of trials picks the states to
  redraw (Voss-McCartney-like pink noise).
* :class:`OUAlphaNoise` -- new: a superposition of Ornstein-Uhlenbeck
  processes (Lorentzians) with log-spaced corner frequencies between
  ``f_min`` and ``f_max`` and weights ``sigma_k^2 ~ tau_k^(alpha - 1)``.
  This is the Dutta-Horn / two-level-fluctuator picture of 1/f noise. It is
  exact in time for any ``dt``, streams forever without buffers, and needs only
  ``O(channels * poles)`` memory, which makes per-cell 1/f fields on a mesh
  affordable. Valid for ``0 < alpha < 2``.

Micromagnetic noise fields
--------------------------
All fields return arrays of shape ``n + (3,)`` in A/m:

* :class:`ThermalField` -- Brown's white thermal field,
  ``sigma = sqrt(2 alpha kB T / (gamma0 mu0 Ms V dt))`` per component.
* :class:`ColouredNoiseField` -- a 1/f^alpha field, either per region (grains,
  defects, masks; with optional Hooge-like ``1/sqrt(volume)`` amplitude
  scaling) or per cell with an optional Gaussian spatial correlation length.
* :class:`AnisotropyNoise` -- 1/f^alpha fluctuations of the local uniaxial
  anisotropy constant (``K -> K + dK(r, t)``), which depend on ``m``.

Analysis helpers: :func:`power_spectral_density`, :func:`fit_spectral_exponent`.
"""

from __future__ import annotations

from math import log10, pi, sqrt
from typing import Callable, Optional, Sequence, Union

import numpy as np

from .config import BOLTZMANN, DTYPE, GAMMA, MU0  # noqa: F401  (BOLTZMANN re-exported)
from .drivers import as_driver


SeedLike = Union[None, int, np.random.Generator]


def _rng(seed: SeedLike) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def _shape(shape) -> tuple:
    if shape is None:
        return ()
    if np.isscalar(shape):
        return (int(shape),)
    return tuple(int(s) for s in shape)


# ---------------------------------------------------------------------------
# Temporal generators
# ---------------------------------------------------------------------------


def kasdin_filter(N: int, alpha: float) -> np.ndarray:
    """Impulse response of the 1/f^alpha fractional integration filter."""
    k = np.arange(1, N, dtype=DTYPE)
    h = np.empty(N, dtype=DTYPE)
    h[0] = 1.0
    h[1:] = np.cumprod((0.5 * alpha + k - 1.0) / k)
    return h


def alpha_noise(
    N: int,
    Qstd: float = 1.0,
    alpha: float = 1.0,
    size=None,
    seed: SeedLike = None,
) -> np.ndarray:
    """Generate ``N`` samples of 1/f^alpha noise (Kasdin's method).

    :param N: number of samples (along the last axis).
    :param Qstd: standard deviation of the driving white noise.
    :param alpha: spectral exponent, PSD ~ 1/f^alpha (0 = white, 2 = Brownian).
    :param size: extra leading shape for independent channels.
    :param seed: seed or ``np.random.Generator``.
    :return: array of shape ``size + (N,)``.
    """
    rng = _rng(seed)
    shape = _shape(size)
    white = rng.normal(0.0, Qstd, shape + (N,))
    h = kasdin_filter(N, alpha)
    # linear convolution through a 2N zero padded real FFT
    L = 2 * N
    out = np.fft.irfft(np.fft.rfft(white, n=L, axis=-1) * np.fft.rfft(h, n=L), n=L, axis=-1)
    return out[..., :N]


class BufferedAlphaNoise:
    """Buffered 1/f^alpha generator (cmtj ``BufferedAlphaNoise``).

    Every ``buffer_size`` ticks a new independent buffer is generated with
    :func:`alpha_noise`. ``shape`` allows many independent channels at once.
    """

    def __init__(
        self,
        buffer_size: int,
        alpha: float,
        std: float,
        scale: float = 1.0,
        shape=None,
        seed: SeedLike = None,
    ):
        self.buffer_size = int(buffer_size)
        self.alpha = float(alpha)
        self.std = float(std)
        self.scale = float(scale)
        self.shape = _shape(shape)
        self.rng = _rng(seed)
        self._h_hat = np.fft.rfft(kasdin_filter(self.buffer_size, self.alpha), n=2 * self.buffer_size)
        self._buffer = None
        self._counter = 0
        self.refills = 0

    def fill_buffer(self) -> np.ndarray:
        L = 2 * self.buffer_size
        white = self.rng.normal(0.0, self.std, self.shape + (self.buffer_size,))
        out = np.fft.irfft(np.fft.rfft(white, n=L, axis=-1) * self._h_hat, n=L, axis=-1)
        self._buffer = np.ascontiguousarray(np.moveaxis(out[..., : self.buffer_size], -1, 0))
        self.refills += 1
        return self._buffer

    def full_buffer(self) -> np.ndarray:
        """Current buffer, time along axis 0 (unscaled)."""
        if self._buffer is None:
            self.fill_buffer()
        return self._buffer

    def tick(self):
        if self._counter == 0:
            self.fill_buffer()
        value = self._buffer[self._counter]
        self._counter = (self._counter + 1) % self.buffer_size
        if self.shape == ():
            return self.scale * float(value)
        return self.scale * value


_AXES = {"all": None, "x": 0, "y": 1, "z": 2}


class VectorAlphaNoise:
    """Vector 1/f^alpha noise (cmtj ``VectorAlphaNoise``).

    :param axis: ``"all"`` for three independent components (normalised to a
        unit vector before scaling, as in cmtj) or ``"x"``, ``"y"``, ``"z"``
        for a single non-normalised component.
    :param shape: extra shape of independent vectors (e.g. number of regions).
    """

    def __init__(
        self,
        buffer_size: int,
        alpha: float,
        std: float,
        scale: float = 1.0,
        axis: str = "all",
        normalized: Optional[bool] = None,
        shape=None,
        seed: SeedLike = None,
    ):
        if axis not in _AXES:
            raise ValueError(f"axis must be one of {list(_AXES)}")
        self.axis = _AXES[axis]
        self.scale = float(scale)
        self.shape = _shape(shape)
        self.normalized = (self.axis is None) if normalized is None else bool(normalized)
        ncomp = 3 if self.axis is None else 1
        self._gen = BufferedAlphaNoise(buffer_size, alpha, std, 1.0, shape=self.shape + (ncomp,), seed=seed)
        self.prev_sample = np.zeros(self.shape + (3,))
        self.current_sample = np.zeros(self.shape + (3,))

    def tick_vector(self) -> np.ndarray:
        raw = self._gen.tick()
        v = np.zeros(self.shape + (3,))
        if self.axis is None:
            v[...] = raw
        else:
            v[..., self.axis] = raw[..., 0]
        if self.normalized:
            norm = np.linalg.norm(v, axis=-1, keepdims=True)
            v = np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)
        self.prev_sample = self.current_sample
        self.current_sample = v
        return v * self.scale

    # cmtj-style alias
    tickVector = tick_vector

    def tick(self):
        v = self.tick_vector()
        return v[..., 0 if self.axis is None else self.axis]


class OneFNoise:
    """cmtj ``OneFNoise``: sum of ``sources`` randomly refreshed uniform states.

    On each tick ``sources`` geometric(``bias``) trials (counting failures,
    as ``std::geometric_distribution``) are drawn; every state hit by a trial
    is redrawn from U(0, 1). Low indices are refreshed often, high indices
    rarely, which produces a ~1/f spectrum. Returns ``scale * sum(states)``.
    """

    def __init__(self, sources: int, bias: float, scale: float = 1.0, shape=None, seed: SeedLike = None):
        self.sources = int(sources)
        self.bias = float(bias)
        self.scale = float(scale)
        self.shape = _shape(shape)
        self.rng = _rng(seed)
        self.state = self.rng.random((self.sources,) + self.shape)
        self._cols = np.arange(int(np.prod(self.shape, dtype=int)))

    def tick(self):
        nch = self._cols.size
        trials = self.rng.geometric(self.bias, size=(self.sources, nch)) - 1
        hit = np.zeros((self.sources + 1, nch), dtype=bool)
        hit[np.minimum(trials, self.sources), self._cols[None, :]] = True
        hit = hit[:-1].reshape(self.state.shape)
        self.state[hit] = self.rng.random(int(hit.sum()))
        out = self.scale * self.state.sum(axis=0)
        return float(out) if self.shape == () else out


class OUAlphaNoise:
    """Streaming 1/f^alpha noise from a superposition of Ornstein-Uhlenbeck processes.

    Each pole ``k`` is an OU process with correlation time ``tau_k`` and
    stationary variance ``s_k^2``; its PSD is a Lorentzian
    ``2 s_k^2 tau_k / (1 + (2 pi f tau_k)^2)``. With ``tau_k`` log-uniformly
    spaced and ``s_k^2 ~ tau_k^(alpha - 1)`` the sum has PSD ~ 1/f^alpha for
    ``f_min << f << f_max`` (flat below ``f_min``, ~1/f^2 above ``f_max``).

    The update ``x <- x exp(-dt/tau) + s sqrt(1 - exp(-2 dt/tau)) xi`` is exact
    for any ``dt``; the process starts in its stationary state.

    :param alpha: spectral exponent, ``0 < alpha < 2``.
    :param std: total standard deviation (rms) of the output.
    :param f_min, f_max: band (Hz) over which the spectrum is 1/f^alpha.
    :param dt: default time step (s).
    :param poles_per_decade: Lorentzians per decade (>= 1; 2-3 gives < 1 dB ripple).
    :param shape: shape of independent channels.
    """

    def __init__(
        self,
        alpha: float,
        std: float,
        f_min: float,
        f_max: float,
        dt: float,
        poles_per_decade: float = 2.0,
        shape=None,
        seed: SeedLike = None,
    ):
        if not 0.0 < alpha < 2.0:
            raise ValueError("OUAlphaNoise supports 0 < alpha < 2")
        if not 0.0 < f_min < f_max:
            raise ValueError("Need 0 < f_min < f_max")
        self.alpha = float(alpha)
        self.std = float(std)
        self.shape = _shape(shape)
        self.rng = _rng(seed)
        decades = log10(f_max / f_min)
        npoles = max(2, int(np.ceil(decades * poles_per_decade)) + 1)
        freqs = np.logspace(log10(f_min), log10(f_max), npoles)
        self.tau = 1.0 / (2 * pi * freqs)
        w = self.tau ** (self.alpha - 1.0)
        self.sigma = self.std * np.sqrt(w / w.sum())
        self._bshape = (npoles,) + (1,) * len(self.shape)
        self.state = self.sigma.reshape(self._bshape) * self.rng.standard_normal((npoles,) + self.shape)
        self._dt = None
        self.set_dt(dt)

    @property
    def n_poles(self) -> int:
        return self.tau.size

    def set_dt(self, dt: float):
        if dt != self._dt:
            self._dt = float(dt)
            decay = np.exp(-self._dt / self.tau)
            self._a = decay.reshape(self._bshape)
            self._b = (self.sigma * np.sqrt(1.0 - decay**2)).reshape(self._bshape)

    def value(self):
        out = self.state.sum(axis=0)
        return float(out) if self.shape == () else out

    def tick(self, dt: Optional[float] = None):
        if dt is not None:
            self.set_dt(dt)
        self.state *= self._a
        self.state += self._b * self.rng.standard_normal(self.state.shape)
        return self.value()

    def theoretical_psd(self, f: np.ndarray) -> np.ndarray:
        """One-sided PSD of the generated process (per channel)."""
        f = np.asarray(f, dtype=DTYPE)[..., None]
        return np.sum(4 * self.sigma**2 * self.tau / (1 + (2 * pi * f * self.tau) ** 2), axis=-1)


def make_temporal_generator(
    kind: str,
    alpha: float,
    std: float,
    dt: float,
    shape,
    seed: SeedLike = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
    poles_per_decade: float = 2.0,
    buffer_size: int = 4096,
):
    """Build a generator with a ``tick()`` returning an array of ``shape``.

    ``kind="ou"`` (default band: ``f_min = 1e-4/dt``, ``f_max = 0.5/dt``)
    or ``kind="buffered"`` (Kasdin buffers of ``buffer_size`` steps, rescaled
    so that the per-sample rms equals ``std`` on average).
    """
    if kind == "ou":
        f_max = 0.5 / dt if f_max is None else f_max
        f_min = 1e-4 / dt if f_min is None else f_min
        return OUAlphaNoise(alpha, std, f_min, f_max, dt, poles_per_decade, shape=shape, seed=seed)
    if kind == "buffered":
        # rms of the Kasdin output for unit white noise, averaged over the buffer
        h = kasdin_filter(buffer_size, alpha)
        rms = sqrt(np.mean(np.cumsum(h**2)))
        return BufferedAlphaNoise(buffer_size, alpha, 1.0, scale=std / rms, shape=shape, seed=seed)
    raise ValueError(f"Unknown generator kind {kind!r}, use 'ou' or 'buffered'")


# ---------------------------------------------------------------------------
# Spatial structure helpers
# ---------------------------------------------------------------------------


def labels_from_masks(masks: Sequence[np.ndarray], n: tuple) -> np.ndarray:
    """Integer label array (0 = no region) from a list of boolean masks.

    Masks may have shape ``n`` or ``n + (3,)`` (as produced by
    :func:`micromagpy.shapes.circular_mask`). Later masks win on overlaps.
    """
    labels = np.zeros(n, dtype=np.int64)
    for i, mask in enumerate(masks, start=1):
        mask = np.asarray(mask, dtype=bool)
        if mask.shape == tuple(n) + (3,):
            mask = mask.any(axis=-1)
        labels[mask.reshape(n)] = i
    return labels


def voronoi_grains(n: tuple, n_grains: int, seed: SeedLike = None, periodic: bool = False) -> np.ndarray:
    """Random Voronoi grain labels (1..n_grains) on the mesh, a simple polycrystal model."""
    rng = _rng(seed)
    n = tuple(int(k) for k in n)
    seeds = rng.random((n_grains, 3)) * np.array(n)
    grid = np.stack(np.meshgrid(*(np.arange(k) + 0.5 for k in n), indexing="ij"), axis=-1)
    d = grid[..., None, :] - seeds
    if periodic:
        d -= np.round(d / np.array(n)) * np.array(n)
    return np.argmin(np.sum(d**2, axis=-1), axis=-1) + 1


def gaussian_correlation_filter(n: tuple, dx: tuple, correlation_length: float):
    """Fourier filter that imposes a Gaussian spatial correlation ``exp(-r^2 / (2 l^2))``.

    Returns ``(kernel_hat, axes)``; filtering is periodic over the mesh and the
    kernel is normalised so that unit-variance white noise stays unit variance.
    """
    axes = tuple(i for i in range(3) if n[i] > 1)
    shape = [n[i] for i in axes]
    ks = np.meshgrid(*(np.fft.fftfreq(n[i], d=dx[i]) * 2 * pi for i in axes), indexing="ij", sparse=True)
    k2 = sum(k**2 for k in ks) if ks else np.zeros(())
    # amplitude filter = sqrt of the PSD of the Gaussian correlation function
    amp = np.exp(-0.25 * correlation_length**2 * k2)
    # normalise: output variance = mean(|amp|^2) for unit white input
    amp = amp / np.sqrt(np.mean(amp**2))
    full = [1, 1, 1]
    for a, s in zip(axes, shape):
        full[a] = s
    return amp.reshape(full + [1]), axes


# ---------------------------------------------------------------------------
# Micromagnetic noise fields
# ---------------------------------------------------------------------------


class ThermalField:
    """Brown's thermal field (white, Gaussian, uncorrelated in space and time).

    ``sigma = sqrt(2 alpha kB T / (gamma0 mu0 Ms V dt))`` per component (A/m),
    with ``gamma0 = mu0 * gamma`` (:data:`micromagpy.config.GAMMA`, m/(A s)).
    Use it as a field that is constant during a step (Stratonovich / Heun).

    :param temperature: K, a callable ``T(t)`` or a :mod:`micromagpy.drivers` driver.
    :param mask: optional boolean array of shape ``n`` (or ``n + (3,)``)
        selecting magnetic cells.
    """

    def __init__(self, n, dx, ms, alpha, temperature, dt, mask=None, seed: SeedLike = None, gamma: float = GAMMA):
        self.n = tuple(int(k) for k in n)
        self.volume = float(np.prod(dx))
        self.ms = float(ms)
        self.alpha = float(alpha)
        self.temperature = as_driver(temperature)
        self.dt = float(dt)
        self.gamma = float(gamma)
        self.rng = _rng(seed)
        self.mask = None
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape == self.n + (3,):
                mask = mask.any(axis=-1)
            self.mask = mask.reshape(self.n + (1,)).astype(DTYPE)

    def sigma(self, t: float = 0.0, dt: Optional[float] = None) -> float:
        T = float(self.temperature(t))
        dt = self.dt if dt is None else dt
        return sqrt(2 * self.alpha * BOLTZMANN * T / (self.gamma * MU0 * self.ms * self.volume * dt))

    def sample(self, t: float = 0.0, m: Optional[np.ndarray] = None, dt: Optional[float] = None) -> np.ndarray:
        h = self.rng.standard_normal(self.n + (3,))
        h *= self.sigma(t, dt)
        if self.mask is not None:
            h *= self.mask
        return h

    __call__ = sample


class ColouredNoiseField:
    """Coloured (1/f^alpha) magnetic field noise on the mesh.

    Spatial structure, pick one:

    * ``regions`` -- integer labels of shape ``n`` (0 = no noise, 1..R = region)
      or a list of boolean masks. Each region carries one independent noise
      vector (e.g. a grain, a defect cluster, a free layer of an MTJ array).
      With ``volume_scaling=True`` the amplitude of each region is scaled by
      ``sqrt(V_ref / V_region)`` (Hooge-like: field noise power ~ 1/volume),
      where ``V_ref`` is the mean region volume.
    * per cell (``regions=None``) -- independent processes in every cell,
      optionally smoothed with a Gaussian ``correlation_length`` (m, periodic
      filtering, rms preserved).

    Orientation: ``axis=None`` gives three independent components (each with
    rms ``amplitude``); a vector ``axis`` gives a scalar process along it.

    :param amplitude: rms field per component, A/m (scalar or per region).
    :param alpha: spectral exponent of the temporal 1/f^alpha spectrum.
    :param dt: simulation time step (s), used by the ``"ou"`` generator.
    :param generator: ``"ou"`` (streaming, default) or ``"buffered"`` (Kasdin, cmtj-like).
    :param generator_kwargs: forwarded to :func:`make_temporal_generator`
        (``f_min``, ``f_max``, ``poles_per_decade``, ``buffer_size``).
    :param mask: optional boolean magnetic-cell mask applied to the result.
    """

    def __init__(
        self,
        n,
        amplitude,
        alpha: float,
        dt: float,
        regions=None,
        axis=None,
        correlation_length: Optional[float] = None,
        dx=None,
        volume_scaling: bool = False,
        generator: str = "ou",
        generator_kwargs: Optional[dict] = None,
        mask=None,
        seed: SeedLike = None,
    ):
        self.n = tuple(int(k) for k in n)
        self.dt = float(dt)
        self.rng = _rng(seed)
        ncomp = 3 if axis is None else 1
        self.axis = None
        if axis is not None:
            u = np.asarray(axis, dtype=DTYPE).reshape(3)
            self.axis = u / np.linalg.norm(u)

        self.labels = None
        if regions is not None:
            if isinstance(regions, np.ndarray) and regions.dtype.kind in "iu" and regions.shape == self.n:
                self.labels = regions.astype(np.int64)
            else:
                self.labels = labels_from_masks(regions, self.n)
            n_regions = int(self.labels.max())
            channel_shape = (n_regions + 1, ncomp)  # row 0 = "no region", zeroed
            amp = np.broadcast_to(np.asarray(amplitude, dtype=DTYPE), (n_regions,)).copy()
            if volume_scaling:
                counts = np.bincount(self.labels.ravel(), minlength=n_regions + 1)[1:].astype(DTYPE)
                ref = counts[counts > 0].mean()
                amp *= np.sqrt(np.divide(ref, counts, out=np.zeros_like(counts), where=counts > 0))
            self._amp = np.concatenate([[0.0], amp])[:, None]
        else:
            if correlation_length is not None and dx is None:
                raise ValueError("dx is required with correlation_length")
            channel_shape = self.n + (ncomp,)
            self._amp = np.asarray(amplitude, dtype=DTYPE)
        self._filter = None
        if regions is None and correlation_length:
            self._filter = gaussian_correlation_filter(self.n, dx, correlation_length)

        self.generator = make_temporal_generator(
            generator, alpha, 1.0, self.dt, channel_shape, seed=self.rng, **(generator_kwargs or {})
        )
        self.mask = None
        if mask is not None:
            mask = np.asarray(mask, dtype=bool)
            if mask.shape == self.n + (3,):
                mask = mask.any(axis=-1)
            self.mask = mask.reshape(self.n + (1,)).astype(DTYPE)
        self._current = None

    def _tick_unit(self) -> np.ndarray:
        if isinstance(self.generator, OUAlphaNoise):
            return self.generator.tick(self.dt)
        return self.generator.tick()

    def tick(self) -> np.ndarray:
        """Advance by one ``dt`` and return the field (A/m)."""
        x = self._tick_unit() * self._amp
        if self.labels is not None:
            x = x[self.labels]  # (n..., ncomp)
        elif self._filter is not None:
            kernel_hat, axes = self._filter
            if axes:
                x = np.real(np.fft.ifftn(np.fft.fftn(x, axes=axes) * kernel_hat, axes=axes))
        h = x * self.axis if self.axis is not None else x
        h = np.ascontiguousarray(h, dtype=DTYPE)
        if self.mask is not None:
            h *= self.mask
        self._current = h
        return h

    def sample(self, t: float = 0.0, m: Optional[np.ndarray] = None, dt: Optional[float] = None) -> np.ndarray:
        return self.tick()

    __call__ = sample

    @property
    def current(self) -> np.ndarray:
        """Last generated field (without advancing)."""
        return self.tick() if self._current is None else self._current


class AnisotropyNoise:
    """1/f^alpha fluctuations of the uniaxial anisotropy, ``K(r, t) = K + dK(r, t)``.

    Returns the extra anisotropy field ``2 dK / (mu0 Ms) (m . u) u``; the
    spatial structure (regions / per cell / correlation length) and generator
    options are the same as for :class:`ColouredNoiseField` with ``dK_std``
    (J/m^3) as the rms amplitude of the scalar fluctuation.
    """

    def __init__(self, n, ms, dK_std, kaxis, alpha: float, dt: float, **kwargs):
        u = np.asarray(kaxis, dtype=DTYPE).reshape(3)
        self.u = u / np.linalg.norm(u)
        self.pref = 2.0 / (MU0 * float(ms))
        # scalar process stored along x of a ColouredNoiseField
        self._dK = ColouredNoiseField(n, dK_std, alpha, dt, axis=(1, 0, 0), **kwargs)

    def delta_k(self) -> np.ndarray:
        """Current dK(r) of shape n (J/m^3)."""
        return self._dK.current[..., 0]

    def tick(self) -> np.ndarray:
        """Advance dK by one time step and return it."""
        return self._dK.tick()[..., 0]

    def field(self, m: np.ndarray) -> np.ndarray:
        """Anisotropy field of the current dK for magnetisation ``m`` (A/m)."""
        dK = self._dK.current[..., :1]
        return self.pref * dK * (m @ self.u)[..., None] * self.u

    def sample(self, t: float = 0.0, m: Optional[np.ndarray] = None, dt: Optional[float] = None) -> np.ndarray:
        if m is None:
            raise ValueError("AnisotropyNoise needs the magnetisation m")
        self.tick()
        return self.field(m)

    __call__ = sample


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def power_spectral_density(x: np.ndarray, dt: float, segments: int = 8, axis: int = 0):
    """One-sided PSD by Welch's method (Hann window, 50% overlap).

    :return: ``(f, psd)`` with ``f > 0`` only (DC dropped).
    """
    x = np.moveaxis(np.asarray(x, dtype=DTYPE), axis, -1)
    N = x.shape[-1]
    nper = max(16, int(2 * N // (segments + 1)))
    step = nper // 2
    win = np.hanning(nper)
    norm = dt / np.sum(win**2)
    starts = range(0, N - nper + 1, step)
    acc = 0.0
    for count, s in enumerate(starts, start=1):
        seg = x[..., s : s + nper]
        seg = seg - seg.mean(axis=-1, keepdims=True)
        acc = acc + np.abs(np.fft.rfft(seg * win, axis=-1)) ** 2
    psd = 2 * norm * acc / count
    f = np.fft.rfftfreq(nper, dt)
    return f[1:], np.moveaxis(psd[..., 1:], -1, axis)


def fit_spectral_exponent(f: np.ndarray, psd: np.ndarray, f_min: float = None, f_max: float = None) -> float:
    """Least-squares fit of ``psd ~ f^-alpha`` in log-log space; returns alpha.

    Fits on log-spaced bins so that the many high-frequency points do not dominate.
    """
    f = np.asarray(f)
    psd = np.asarray(psd)
    sel = np.ones_like(f, dtype=bool)
    if f_min is not None:
        sel &= f >= f_min
    if f_max is not None:
        sel &= f <= f_max
    if sel.sum() < 2:
        raise ValueError("Fewer than 2 frequency points in the fit range; use a longer signal")
    lf, lp = np.log10(f[sel]), np.log10(psd[sel])
    edges = np.linspace(lf.min(), lf.max(), 25)
    idx = np.digitize(lf, edges)
    xs = [lf[idx == i].mean() for i in np.unique(idx)]
    ys = [lp[idx == i].mean() for i in np.unique(idx)]
    slope = np.polyfit(xs, ys, 1)[0]
    return -slope


NoiseSource = Union[ThermalField, ColouredNoiseField, AnisotropyNoise, Callable]
