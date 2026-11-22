# Curated examples

Micromagnetic counterparts of the [cmtj curated examples](https://github.com/LemurPwned/cmtj/tree/master/curated-examples),
grouped by topic. Every folder holds the scripts and a `results/` directory
with the figures, one page per example and `timings.json` (wall times).
Run a script from anywhere, e.g. `python dynamics/fmr_basic.py`; outputs land in
the `results/` next to it.

Scripts marked **JAX** use the optional JAX backend
([`micromagpy/jax_backend.py`](../../micromagpy/jax_backend.py)) to integrate
hundreds of independent simulations in one batch, which is fast on a GPU
(`pip install "micromagpy[jax]"`, or `jax[cuda13]` for NVIDIA). The others use
the numba path.

## [`noise/`](noise)

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| [`noise_spectra.py`](noise/results/noise_spectra.md) | `cmtj.models.noise` plots | PSD (with fitted exponent), autocorrelation and traces of every 1/f generator |
| [`seeded_noise_demo.py`](noise/results/seeded_noise_demo.md) | `seeded-noise-demo.py` | same seed gives a bit-identical thermal + 1/f trajectory |
| [`domain_noise.py`](noise/results/domain_noise.md) | (new) | thermal <m> noise of an element made of N bars with dispersed PMA / IMA anisotropy; spectra vs N and sigma, GIFs of m_x, m_y, m_z |
| [`noise_spectrogram.py`](noise/results/noise_spectrogram.md) | (new) | spectra driven only by thermal + 1/f grain noise vs field; the thermal FMR line follows Kittel, the 1/f band sinks under the thermal floor as the field stiffens m |

## [`dynamics/`](dynamics)

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| [`fmr_basic.py`](dynamics/results/fmr_basic.md) | `fmr-basic.py` | pulse-excited FMR field scan of a PMA element with hysteresis (relax + RK45) |
| [`sto.py`](dynamics/results/sto.md) | `sto.py` | STT oscillator: trajectory and f(j) vs the macrospin law f = gamma0 aJ P / (4 pi alpha) |

## [`switching/`](switching)

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| [`switching_hysteresis.py`](switching/results/switching_hysteresis.md) | `switching-hysteresis.py` | field-driven M(H) loop of an in-plane disk, coercive field vs Stoner-Wohlfarth |
| [`sot_switching.py`](switching/results/sot_switching.md) | `cims-hysteresis.py` | SOT switching loops of a PMA element; polarity reverses with H_x, field-like torque lowers the switching current |
| [`sot_phase_diagram.py`](switching/results/sot_phase_diagram.md) **JAX** | `cims-stability-stability-diagram.py` | switching probability over (H_x, j) at 300 K from thermal replicas |
| [`p_bit.py`](switching/results/p_bit.md) **JAX** | `p-bit.py` | telegraph noise of a low-barrier PMA disk and the p-bit sigmoid <m_z>(H) = tanh(mu0 Ms V H / kT) |

## [`transport/`](transport)

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| [`magnetoresistance.py`](transport/results/magnetoresistance.md) | `harmonic-hall.py` | Rxx / Rxy field sweeps (AMR, SMR, AHE, planar Hall) of a PMA Hall bar |
| [`spin_diode.py`](transport/results/spin_diode.md) **JAX** | `vsd-basic-field-scan.py` | spin-diode mixing voltage vs field at several frequencies, resonance vs the Kittel law |

Not ported: `afm-resonance.py`, `iec-coupling.py` and the three-layer part of
`fmr-basic.py` need interlayer (RKKY) or two-sublattice coupling, which
micromagpy does not model yet; `ahe_loops.py` is covered by
`switching_hysteresis.py` + `magnetoresistance.py`, and
`vsd-basic-freq-scan.py` by `spin_diode.py`.

## Timings

Wall times from each folder's `results/timings.json`. numba runs: older entries on a 4-core container, `switching_hysteresis` / `magnetoresistance` on a Mac M5 Pro with one numba thread. JAX runs on the NVIDIA GB10 (DGX Spark).

| Example | Mesh | Work | Backend | Wall time |
|---|---|---|---|---|
| noise/noise_spectra | – | 2²⁰ samples × 6 generators | numpy | 2.5 s |
| noise/seeded_noise_demo | 16×16×1 | 15 000 Heun steps | numba | 3.4 s |
| noise/noise_spectrogram | 16×16×1 | 11 fields × 320 000 Heun steps | numba | 857 s |
| noise/domain_noise (GIFs) | 32×8×1 | 100 frames, per case | numba | 12–16 s |
| noise/domain_noise (sweeps) | 32×8×1 | 36 runs × 6 ns, per case | numba | 583–609 s |
| dynamics/fmr_basic | 16×16×1 | 41 fields, 328k RK45 steps | numba | 213 s |
| dynamics/sto | 16×16×1 | 8 currents × 6 ns | numba | 96 s |
| switching/sot_switching | 16×16×1 | 3 loops × 40 pulses | numba | 287 s |
| switching/switching_hysteresis | 10×10×1 | 480 relaxed points | numba (1 thread) | 22 s |
| switching/p_bit | 4×4×1 | 168 trajectories × 10⁶ steps | JAX f64, GB10 | 107 s (numba est. 7.4×10⁴ s) |
| switching/sot_phase_diagram | 16×16×1 | 5000 runs × 20 000 steps | JAX f32, GB10 | 242 s (numba est. ~8 h) |
| transport/magnetoresistance | 32×32×1 | 400 relaxed points | numba (1 thread) | 89 s |
| transport/spin_diode | 32×32×1 | 600 runs × 200 000 RK4 steps | JAX f32, GB10 | 1409 s |
