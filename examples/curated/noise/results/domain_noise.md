# Thermal noise of domains with dispersed PMA / IMA anisotropy

Script: [`../domain_noise.py`](../domain_noise.py) · new example

## Setup
- **Element:** 160×40×2 nm (32×8×1 cells, Ms = 800 kA/m, A = 13 pJ/m, α = 0.05), split into N bars ("domains") along x.
- **Per-bar anisotropy:** bar i has its own magnitude and axis.
  - Magnitude: K_i = K0·(1 + σ·ξ_i), with ξ ~ N(0, 1), clipped at 0.05·K0.
  - Axis: tilted by an angle ~ N(0, σ) rad from the nominal direction.
  - **PMA:** K0 = 800 kJ/m³ around z, so K_eff ≈ 400 kJ/m³; the tilt azimuth is random.
  - **IMA:** K0 = 50 kJ/m³ around x, in-plane.
  - **Mixed:** PMA and IMA bars alternate.
- **Protocol:**
  1. relax with the energy minimiser;
  2. 6 ns with a white thermal field in every cell (same global T), stochastic Heun with dt = 0.1 ps;
  3. record S(f) = Σ_c S_⟨m_c⟩(f), the PSD of the total ⟨m⟩ noise.
- **Sweep:** N ∈ {1, 2, 4, 8} × σ ∈ {0, 0.2, 0.5}, each point averaged over 3 disorder realisations, at **300 K** and **50 K**.

Regenerate:

```bash
python domain_noise.py sweep --temperature 300
python domain_noise.py sweep --temperature 50
python domain_noise.py gif --temperature 300
python domain_noise.py gif --temperature 50
python domain_noise.py compare --compare 300,50
```

## Spectra vs number of domains and dispersion
Each figure has four panels:
1. PSD vs N at σ = 0.5;
2. PSD vs σ at N = 8;
3. spectral width;
4. integrated noise ⟨δm²⟩, normalised to N = 1, σ = 0.

### PMA, 300 K
![PMA 300 K](domain-noise-pma-T300.png)

### PMA, 50 K
![PMA 50 K](domain-noise-pma-T50.png)

### IMA, 300 K
![IMA 300 K](domain-noise-ima-T300.png)

### IMA, 50 K
![IMA 50 K](domain-noise-ima-T50.png)

Each cell below gives spectral centroid / spectral width / integrated noise relative to (N = 1, σ = 0).

**PMA, 300 K**

| N \ σ | 0.0 | 0.2 | 0.5 |
| --- | --- | --- | --- |
| 1 | 29.3 GHz / 6.2 GHz / 1.00 | 32.7 GHz / 6.2 GHz / 1.07 | 37.3 GHz / 8.6 GHz / 0.72 |
| 2 | 29.3 GHz / 6.2 GHz / 1.00 | 24.2 GHz / 10.2 GHz / 1.22 | 7.1 GHz / 9.6 GHz / 6.70 |
| 4 | 29.3 GHz / 6.2 GHz / 1.00 | 24.6 GHz / 9.4 GHz / 1.31 | 15.4 GHz / 12.1 GHz / 2.31 |
| 8 | 29.3 GHz / 6.2 GHz / 1.00 | 22.2 GHz / 10.5 GHz / 1.46 | 15.8 GHz / 12.7 GHz / 1.92 |

**PMA, 50 K**

| N \ σ | 0.0 | 0.2 | 0.5 |
| --- | --- | --- | --- |
| 1 | 30.9 GHz / 6.3 GHz / 1.00 | 34.2 GHz / 6.1 GHz / 1.12 | 39.2 GHz / 8.5 GHz / 0.78 |
| 2 | 30.9 GHz / 6.3 GHz / 1.00 | 25.2 GHz / 10.1 GHz / 1.38 | 7.0 GHz / 9.9 GHz / 7.11 |
| 4 | 30.9 GHz / 6.3 GHz / 1.00 | 25.9 GHz / 9.6 GHz / 1.34 | 15.7 GHz / 12.5 GHz / 2.41 |
| 8 | 30.9 GHz / 6.3 GHz / 1.00 | 23.7 GHz / 10.9 GHz / 1.37 | 16.7 GHz / 12.7 GHz / 1.96 |

**IMA, 300 K**

| N \ σ | 0.0 | 0.2 | 0.5 |
| --- | --- | --- | --- |
| 1 | 10.9 GHz / 3.9 GHz / 1.00 | 11.0 GHz / 3.9 GHz / 0.98 | 11.2 GHz / 3.9 GHz / 0.98 |
| 2 | 10.9 GHz / 3.9 GHz / 1.00 | 10.9 GHz / 3.9 GHz / 0.97 | 10.1 GHz / 4.2 GHz / 1.01 |
| 4 | 10.9 GHz / 3.9 GHz / 1.00 | 11.0 GHz / 3.9 GHz / 1.04 | 10.1 GHz / 4.2 GHz / 1.13 |
| 8 | 10.9 GHz / 3.9 GHz / 1.00 | 10.7 GHz / 3.9 GHz / 1.08 | 9.5 GHz / 3.9 GHz / 1.42 |

**IMA, 50 K**

| N \ σ | 0.0 | 0.2 | 0.5 |
| --- | --- | --- | --- |
| 1 | 11.4 GHz / 3.9 GHz / 1.00 | 11.5 GHz / 3.9 GHz / 0.98 | 11.7 GHz / 3.8 GHz / 0.98 |
| 2 | 11.4 GHz / 3.9 GHz / 1.00 | 11.5 GHz / 3.9 GHz / 0.97 | 10.8 GHz / 4.2 GHz / 1.02 |
| 4 | 11.4 GHz / 3.9 GHz / 1.00 | 11.5 GHz / 3.9 GHz / 0.99 | 10.6 GHz / 4.3 GHz / 1.06 |
| 8 | 11.4 GHz / 3.9 GHz / 1.00 | 11.2 GHz / 3.9 GHz / 1.01 | 9.8 GHz / 4.0 GHz / 1.34 |

### Observations
- **σ = 0:** the spectrum is identical for every N, as it must be. The bars are then the same material and exchange-coupled.
- **PMA is very sensitive to anisotropy dispersion once there are several domains.** At N = 4–8 and σ = 0.5:
  - the spectral centroid falls from ~30 to ~16 GHz;
  - the width roughly doubles (6 → 12–13 GHz);
  - the integrated noise grows ~1.9–2.0×.
- **Cause of the PMA change:** bars whose K drops below μ0Ms²/2 lose their perpendicular stiffness. They become soft, effectively in-plane, regions that add strong low-frequency noise.
- **The N = 2, σ = 0.5 outlier:** in one disorder realisation half of the element goes soft, which gives a 6.7× rise at 300 K and 7.1× at 50 K.
- **IMA barely reacts:** the in-plane shape anisotropy dominates the small IMA K. The width stays ~4 GHz, and the integrated noise changes by at most ~1.4× (N = 8, σ = 0.5), with the FMR line slightly red-shifted.

## 300 K vs 50 K
![300 K vs 50 K](domain-noise-T300-vs-T50.png)

- **Linear regime:** for a linear (small-fluctuation) thermal response, S(f)/T is independent of temperature and the noise ratio between 300 K and 50 K is 6.
- **All configurations follow it:** the measured ratio is 5.5–6.7 for PMA and 5.9–6.4 for IMA, and the T-normalised spectra overlap (left panels).
- **Thermal softening:** the FMR peak at 300 K sits slightly below the 50 K one, because thermal fluctuations reduce the effective anisotropy.
- **PMA with N = 2 and σ = 0.5** gives 5.9 here. An earlier run (numba path, other thermal draws) gave 11.8 for this case. The soft-bar case is realisation dependent, so more seeds are needed before reading a non-linear response into it.

## Animations
N = 4 bars, σ = 0.5. Each GIF shows:
- the m_x, m_y and m_z maps with bar boundaries and each bar's anisotropy type;
- per-bar ⟨m_z⟩ traces;
- the noise spectrum. The red line is the running Welch estimate (1 ns windows) from the data so far, converging to the full-run element spectrum (dashed). Faint lines are the per-bar spectra.

| | 300 K | 50 K |
| --- | --- | --- |
| PMA bars | ![](domain-noise-pma-T300.gif) | ![](domain-noise-pma-T50.gif) |
| IMA bars | ![](domain-noise-ima-T300.gif) | ![](domain-noise-ima-T50.gif) |
| mixed PMA / IMA | ![](domain-noise-mixed-T300.gif) | ![](domain-noise-mixed-T50.gif) |

## Timings (JAX float64, Mac M5 Pro CPU)
| run | wall time | former numba path (4-core container) |
| --- | --- | --- |
| GIF, per case (100 frames) | 2.6–3.4 s | 12–16 s |
| sweep, per mode and temperature (36 runs × 6 ns) | 134–182 s | 583–609 s |
