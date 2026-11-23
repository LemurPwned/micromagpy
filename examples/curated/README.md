# Curated examples

Micromagnetic counterparts of the [cmtj curated examples](https://github.com/LemurPwned/cmtj/tree/master/curated-examples),
grouped by topic. Every folder holds the scripts and a `results/` directory
with the figures, one page per example and `timings.json` (wall times).
Run a script from anywhere, e.g. `python dynamics/fmr_basic.py`; outputs land in
the `results/` next to it.

Scripts marked **JAX** use the optional JAX backend
([`micromagpy/jax_backend.py`](../../micromagpy/jax_backend.py)) to integrate
hundreds of independent simulations in one batch, which is fast on a GPU
(`pip install "micromagpy[cuda]"` for NVIDIA). All examples now use JAX; the
hysteresis, magnetoresistance, FMR, STO, SOT-loop and noise examples were
ported from the former numba path, whose wall times are kept in brackets.

## [`noise/`](noise)

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| [`noise_spectra.py`](noise/results/noise_spectra.md) | `cmtj.models.noise` plots | PSD (with fitted exponent), autocorrelation and traces of every 1/f generator |
| [`seeded_noise_demo.py`](noise/results/seeded_noise_demo.md) **JAX** | `seeded-noise-demo.py` | same seed gives a bit-identical thermal + 1/f trajectory |
| [`domain_noise.py`](noise/results/domain_noise.md) **JAX** | (new) | thermal <m> noise of an element made of N bars with dispersed PMA / IMA anisotropy; spectra vs N and sigma, GIFs of m_x, m_y, m_z |
| [`noise_spectrogram.py`](noise/results/noise_spectrogram.md) **JAX** | (new) | spectra driven only by thermal + 1/f grain noise vs field; the thermal FMR line follows Kittel, the 1/f band sinks under the thermal floor as the field stiffens m |

## [`dynamics/`](dynamics)

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| [`fmr_basic.py`](dynamics/results/fmr_basic.md) **JAX** | `fmr-basic.py` | pulse-excited FMR field scan of a PMA element with hysteresis (relax + RK45) |
| [`sto.py`](dynamics/results/sto.md) **JAX** | `sto.py` | STT oscillator: trajectory and f(j) vs the macrospin law f = gamma0 aJ P / (4 pi alpha) |

## [`switching/`](switching)

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| [`switching_hysteresis.py`](switching/results/switching_hysteresis.md) **JAX** | `switching-hysteresis.py` | field-driven M(H) loop of an in-plane disk, coercive field vs Stoner-Wohlfarth |
| [`sot_switching.py`](switching/results/sot_switching.md) **JAX** | `cims-hysteresis.py` | SOT switching loops of a PMA element; polarity reverses with H_x, field-like torque lowers the switching current |
| [`sot_phase_diagram.py`](switching/results/sot_phase_diagram.md) **JAX** | `cims-stability-stability-diagram.py` | switching probability over (H_x, j) at 300 K from thermal replicas |
| [`p_bit.py`](switching/results/p_bit.md) **JAX** | `p-bit.py` | telegraph noise of a low-barrier PMA disk and the p-bit sigmoid <m_z>(H) = tanh(mu0 Ms V H / kT) |

## [`transport/`](transport)

| example | cmtj counterpart | what it shows |
| --- | --- | --- |
| [`magnetoresistance.py`](transport/results/magnetoresistance.md) **JAX** | `harmonic-hall.py` | Rxx / Rxy field sweeps (AMR, SMR, AHE, planar Hall) of a PMA Hall bar |
| [`spin_diode.py`](transport/results/spin_diode.md) **JAX** | `vsd-basic-field-scan.py` | spin-diode mixing voltage vs field at several frequencies, resonance vs the Kittel law |

Not ported: `afm-resonance.py`, `iec-coupling.py` and the three-layer part of
`fmr-basic.py` need interlayer (RKKY) or two-sublattice coupling, which
micromagpy does not model yet; `ahe_loops.py` is covered by
`switching_hysteresis.py` + `magnetoresistance.py`, and
`vsd-basic-freq-scan.py` by `spin_diode.py`.

## Macrospin comparison

[`compare/`](compare/README.md) runs cmtj on the device of each ported example and puts the macrospin next to the micromagnetic result: hysteresis, magnetoresistance, FMR, STO, SOT switching, the SOT phase diagram, the p-bit and the spin diode. It shows where a macrospin is enough and where the mesh matters.

## Timings

Wall times from each folder's `results/timings.json`. JAX runs on the Mac M5 Pro CPU or the NVIDIA GB10 (DGX Spark). The former numba times are kept in brackets (noise examples on a 4-core container, the others on the M5 Pro with one thread).

| Example | Mesh | Work | Backend | Wall time |
|---|---|---|---|---|
| noise/noise_spectra | – | 2²⁰ samples × 6 generators | numpy | 2.5 s |
| noise/seeded_noise_demo | 16×16×1 | 15 000 Heun steps | JAX f64, Mac CPU (numba) | 2.0 s (3.4 s) |
| noise/noise_spectrogram | 16×16×1 | 11 fields × 320 000 Heun steps (vmap) | JAX f64, Mac CPU (numba) | 259 s (857 s) |
| noise/domain_noise (GIFs) | 32×8×1 | 100 frames, per case | JAX f64, Mac CPU (numba) | 2.6–3.4 s (12–16 s) |
| noise/domain_noise (sweeps) | 32×8×1 | 36 runs × 6 ns, per case | JAX f64, Mac CPU (numba) | 134–182 s (583–609 s) |
| dynamics/fmr_basic | 16×16×1 | 100 fields, 800k RK45 steps | JAX f64, Mac CPU / GB10 (numba 1 thread) | 156 / 286 s (803 s) |
| dynamics/sto | 16×16×1 | 25 currents × 6 ns (vmap) | JAX f64, GB10 (8 currents: Mac CPU 17 s, numba 90 s) | 8.7 s |
| switching/sot_switching | 16×16×1 | 3 loops × 261 pulses (vmap) | JAX f64, Mac CPU / GB10 (numba 1 thread) | 548 / 251 s (3093 s) |
| switching/switching_hysteresis | 10×10×1 | 480 relaxed points | JAX f64, Mac CPU / GB10 (numba 1 thread) | 4.0 / 14.5 s (22 s) |
| switching/p_bit | 4×4×1 | 168 trajectories × 10⁶ steps | JAX f64, GB10 | 107 s (numba est. 7.4×10⁴ s) |
| switching/sot_phase_diagram | 16×16×1 | 20 000 runs × 20 000 steps | JAX f32, GB10 | 1049 s |
| transport/magnetoresistance | 32×32×1 | 400 relaxed points | JAX f64, Mac CPU / GB10 (numba 1 thread) | 71 / 42 s (89 s) |
| transport/spin_diode | 32×32×1 | 600 runs × 200 000 RK4 steps | JAX f32, GB10 | 1409 s |
