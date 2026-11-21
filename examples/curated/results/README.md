# Curated example results

One page per example with figures, short descriptions and key numbers. Scripts live one directory up; regenerate everything with e.g. `python ../fmr_basic.py`. Wall times are in [`timings.json`](timings.json) and were measured on a 4-core cloud container.

| page | what it shows | figure |
| --- | --- | --- |
| [noise_spectra](noise_spectra.md) | PSD, autocorrelation and throughput of every 1/f^α generator | [png](noise-spectra.png) |
| [seeded_noise_demo](seeded_noise_demo.md) | bit-identical reruns of a thermal + 1/f simulation with the same seed | [png](seeded-noise-demo.png) |
| [fmr_basic](fmr_basic.md) | pulse-excited FMR field scan with hysteresis (relax + RK45) | [png](fmr-basic.png) |
| [noise_spectrogram](noise_spectrogram.md) | thermal FMR line and 1/f band vs field, noise-driven only | [png](noise-spectrogram.png) |
| [sto](sto.md) | spin-transfer oscillator, f(j) vs the macrospin law | [png](sto.png) |
| [sot_switching](sot_switching.md) | SOT switching loops, polarity vs bias field, field-like torque | [png](sot-switching.png) |
| [domain_noise](domain_noise.md) | thermal noise of PMA / IMA / mixed bar domains vs number of domains and anisotropy dispersion, at 300 K and 50 K, with GIFs | [PMA](domain-noise-pma-T300.png) · [IMA](domain-noise-ima-T300.png) |
