# Spin-transfer oscillator

Script: [`../sto.py`](../sto.py) · cmtj counterpart: `sto.py` · wall time JAX 17 s (Mac CPU) / 6.1 s (GB10), numba 90 s

The device is a 64×64×2 nm in-plane free layer (Ms = 800 kA/m, α = 0.02) with a perpendicular polariser p = z. The torque is cmtj-style Slonczewski STT with P = 0.5 and Λ = 1, so η = P/2. Each current runs for 6 ns with adaptive RK45.

Above threshold, the damping-like torque balances damping on an out-of-plane precession orbit. The macrospin limit then gives m_z = a_J·η/(α·Ms) and a frequency linear in current, f = γ0·a_J·P/(4πα).

![sto](sto.png)

Panels:
- **left:** trajectory of ⟨m⟩ on the unit sphere;
- **middle:** time traces;
- **right:** frequency (from zero crossings of ⟨m_x⟩) vs current, with |⟨m⟩| as a uniformity measure.

| j (10¹⁰ A/m²) | 1.0 | 1.9 | 2.7 | 3.6 | 4.4 | 5.3 | 6.1 | 7.0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| f micromagnetic (GHz) | 1.50 | 2.50 | 3.67 | 4.17 | 5.00 | 5.17 | 5.17 | 7.17 |
| f macrospin (GHz) | 1.44 | 2.67 | 3.91 | 5.14 | 6.38 | 7.61 | 8.85 | 10.08 |
| \|⟨m⟩\| | 0.998 | 0.997 | 0.995 | 0.988 | 0.970 | 0.949 | 0.928 | 0.961 |

The micromagnetic frequency follows the macrospin law while precession stays uniform. Above ~3×10¹⁰ A/m² the precession becomes non-uniform and the frequency falls below the macrospin line.
