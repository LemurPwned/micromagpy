# Curated examples

Micromagnetic counterparts of the [cmtj curated examples](https://github.com/LemurPwned/cmtj/tree/master/curated-examples),
focused on noise and spectra. Run from this directory, e.g. `python fmr_basic.py`.
Figures, GIFs and a short write-up for every example are in [`results/`](results/README.md); wall times in `results/timings.json`.

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| `noise_spectra.py` | `cmtj.models.noise` plots | PSD (with fitted exponent), autocorrelation and traces of every 1/f generator |
| `seeded_noise_demo.py` | `seeded-noise-demo.py` | same seed gives a bit-identical thermal + 1/f trajectory |
| `fmr_basic.py` | `fmr-basic.py` | pulse-excited FMR field scan of a PMA element with hysteresis (relax + RK45) |
| `sto.py` | `sto.py` | STT oscillator (in-plane free layer, perpendicular polariser): trajectory and f(j) vs the macrospin law f = gamma0 aJ P / (4 pi alpha) |
| `sot_switching.py` | `cims-hysteresis.py` | SOT switching loops of a PMA element; polarity reverses with H_x, field-like torque lowers the switching current |
| `domain_noise.py` | (new) | thermal <m> noise of an element made of N bars with dispersed PMA / IMA anisotropy (magnitude and axis, sigma); spectra vs N and sigma, GIFs of m_x, m_y, m_z |
| `noise_spectrogram.py` | (new) | spectra driven only by thermal + 1/f grain noise vs field (stochastic Heun); the thermal FMR line follows Kittel, the 1/f band (fit α = 1.02 at H = 0) is pushed under the thermal floor as the field stiffens m |

See [`results/README.md`](results/README.md) for all figures and animations.

## Timings

| run | wall time |
| --- | --- |
| `domain_noise_gif_ima_T300` | 15.16 s |
| `domain_noise_gif_ima_T50` | 12.06 s |
| `domain_noise_gif_mixed_T300` | 14.84 s |
| `domain_noise_gif_mixed_T50` | 11.9 s |
| `domain_noise_gif_pma_T300` | 15.5 s |
| `domain_noise_gif_pma_T50` | 12.23 s |
| `domain_noise_sweep_ima_T300` | 609.32 s |
| `domain_noise_sweep_ima_T50` | 583.12 s |
| `domain_noise_sweep_pma_T300` | 604.71 s |
| `domain_noise_sweep_pma_T50` | 596.08 s |
| `fmr_basic` | 213.38 s |
| `noise_spectra` | 2.53 s |
| `noise_spectrogram` | 857.07 s |
| `seeded_noise_demo` | 3.39 s |
| `sot_switching` | 286.65 s |
| `sto` | 96.09 s |

Measured on a 4-core cloud container, one job at a time. Details per example in [`results/`](results/README.md).
