70 lines of NumPy
=================

This code solves the MuMag standard problem #4 in less than 70 lines of pure
Python without counting empty lines and comments :). The code makes largely
use of the NumPy library and computes the exchange field by finite differences
and the demagnetization field with a fast convolution algorithm. Since the
magnetization in finite-difference micromagnetics is represented by a
multi-dimensional array and the NumPy library features a rich interface for
this data structure, the presented code is a good starting point for the
development of novel algorithms.

A detailed description of the code can be found here: http://arxiv.org/abs/1411.7188

micromagpy
==========

`micromagpy` grew out of the code above into a small package (`pip install -e .`,
tests: `pytest tests`). The demag tensor, exchange/anisotropy fields and LLG
updates are numba kernels; the demag convolution uses real FFTs padded to fast
sizes.

Noise (1/f and thermal)
-----------------------

`micromagpy.noise` ports the noise generators of
[cmtj](https://github.com/LemurPwned/cmtj) and extends them to meshes:

| generator | what it is |
| --- | --- |
| `alpha_noise`, `BufferedAlphaNoise` | Kasdin fractional-integration filter, PSD ~ 1/f^alpha (cmtj `BufferedAlphaNoise`) |
| `VectorAlphaNoise` | 3-component buffered noise, normalised like cmtj |
| `OneFNoise` | cmtj's geometric-trial pink noise |
| `OUAlphaNoise` | streaming 1/f^alpha from log-spaced Ornstein-Uhlenbeck processes (0 < alpha < 2), exact for any dt, O(poles) memory per channel |

Micromagnetic noise sources (fields in A/m, shape `n + (3,)`):

- `ThermalField` - Brown's white thermal field, `sqrt(2 alpha kB T / (gamma0 mu0 Ms V dt))`.
- `ColouredNoiseField` - 1/f^alpha field per region (labels, masks or
  `voronoi_grains`, optional Hooge-like 1/sqrt(volume) scaling) or per cell
  with an optional Gaussian correlation length; optional fixed axis.
- `AnisotropyNoise` - 1/f^alpha fluctuations of the local uniaxial anisotropy.

```python
from micromagpy.noise import ColouredNoiseField, ThermalField, voronoi_grains
from micromagpy.simulation import simulate

grains = voronoi_grains(n, 30, seed=0)
noise = [
    ColouredNoiseField(n, 2e3, alpha=1.0, dt=dt, regions=grains, volume_scaling=True),
    ThermalField(n, dx, ms, damping, 300.0, dt),
]
res = simulate(m, dx, ms, A, damping, dt, steps, K=K, noise=noise)  # stochastic Heun
```

Analysis helpers: `power_spectral_density` (Welch) and `fit_spectral_exponent`.
See `examples/noise_sim.py` for a full example.
