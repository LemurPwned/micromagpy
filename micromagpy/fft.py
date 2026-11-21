"""Thin FFT wrapper: prefer multithreaded ``scipy.fft``, fall back to NumPy."""

import os

try:  # pragma: no cover - depends on the environment
    from scipy import fft as _fft

    _WORKERS = int(os.environ.get("MICROMAGPY_FFT_WORKERS", "-1"))

    def rfftn(x, s=None, axes=None):
        return _fft.rfftn(x, s=s, axes=axes, workers=_WORKERS)

    def irfftn(x, s=None, axes=None):
        return _fft.irfftn(x, s=s, axes=axes, workers=_WORKERS)

    def next_fast_len(k: int) -> int:
        return _fft.next_fast_len(k, real=True)

except ImportError:  # pragma: no cover
    import numpy as _np

    def rfftn(x, s=None, axes=None):
        return _np.fft.rfftn(x, s=s, axes=axes)

    def irfftn(x, s=None, axes=None):
        return _np.fft.irfftn(x, s=s, axes=axes)

    def next_fast_len(k: int) -> int:
        """Smallest 2^a 3^b 5^c >= k."""
        best = 1 << max(int(k) - 1, 0).bit_length()
        p5 = 1
        while p5 < best:
            p35 = p5
            while p35 < best:
                p = p35
                while p < k:
                    p *= 2
                best = min(best, p)
                p35 *= 3
            p5 *= 5
        return best


def fft_axes(n) -> tuple:
    """Spatial axes that carry more than one cell (the only ones transformed)."""
    return tuple(i for i in range(3) if n[i] > 1)


def padded_shape(n, fast: bool = True) -> tuple:
    """Zero-padded grid used for the FFT convolution.

    Any size >= 2n - 1 gives the exact (non-periodic) convolution on the first
    n cells, so by default we round up to a size with small prime factors,
    which is much faster for the FFT (e.g. 2 * 100 - 1 = 199 is prime -> 200).
    Axes with a single cell are not padded.
    """
    if not fast:
        return tuple(2 * int(k) - 1 for k in n)
    return tuple(next_fast_len(2 * int(k) - 1) if int(k) > 1 else 1 for k in n)
