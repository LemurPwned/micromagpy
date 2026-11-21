# Curated examples

Micromagnetic counterparts of the [cmtj curated examples](https://github.com/LemurPwned/cmtj/tree/master/curated-examples),
focused on noise and spectra. Run from this directory, e.g. `python fmr_basic.py`.
Figures are written to `figures/`, wall times to `figures/timings.json`.

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| `noise_spectra.py` | `cmtj.models.noise` plots | PSD (with fitted exponent), autocorrelation and traces of every 1/f generator |
| `seeded_noise_demo.py` | `seeded-noise-demo.py` | same seed gives a bit-identical thermal + 1/f trajectory |
| `fmr_basic.py` | `fmr-basic.py` | pulse-excited FMR field scan of a PMA element with hysteresis (relax + RK45) |
| `noise_spectrogram.py` | (new) | spectra driven only by thermal + 1/f grain noise vs field (stochastic Heun); the thermal FMR line follows Kittel, the 1/f band (fit α = 1.02 at H = 0) is pushed under the thermal floor as the field stiffens m |

![noise spectra](figures/noise-spectra.png)
![seeded noise](figures/seeded-noise-demo.png)
![fmr](figures/fmr-basic.png)
![noise spectrogram](figures/noise-spectrogram.png)

## Timings

| example | mesh | work | solver | wall time |
| --- | --- | --- | --- | --- |
| noise_spectra | - | 1,048,576 samples x 6 generators | - | 2.53 s |
| seeded_noise_demo | 16x16x1 | 3 runs x 0.5 ns (15,000 steps) | heun | 3.39 s |
| fmr_basic | 16x16x1 | 41 fields x 4 ns (327,954 RK45 steps) | relax + rk45 | 213.38 s |
| noise_spectrogram | 16x16x1 | 11 fields x 16 ns (3,520,000 steps) | heun (dt = 50 fs) | 857.07 s |

Generator throughput (16 channels per tick):

| generator | M samples/s |
| --- | --- |
| Kasdin buffered, α=1 | 8.31 |
| OU sum, α=0.5 | 2.79 |
| OU sum, α=1 | 2.81 |
| OU sum, α=1.5 | 2.72 |
| OneF (cmtj), 20 sources | 1.12 |
| VectorAlpha x-comp, α=1 | 3.16 |

Measured on a 4-core cloud container (numba parallel + scipy.fft workers).

