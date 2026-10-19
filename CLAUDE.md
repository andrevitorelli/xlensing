# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Overview

`xlensing` measures, models and simulates the weak-lensing excess surface density ΔΣ(R) of galaxy clusters. See `README.md` (features, speed) and `docs/MANUAL.md` (usage). Install with `pip install -e ".[test]"`; run `pytest` (≈130 tests) and `python scripts/benchmark.py`.

## Modules

- `cosmo.py` — Planck18 defaults; `DA`, `H`, `rhocz`, `rhoM` (note: `rhoM(z)` = Ω_m,0 ρ_crit(z), see its docstring), constants.
- `data.py` — measurement: `metacal_cluster_lensing` / `lensfit_cluster_lensing` (per-galaxy quantities, no background cut), then `signal`, `bootstrap_signal`, `stacked_signal`, `single_cluster` on a (6, N) "stake" with rows Σ_crit, e_t, e_x, W, R, M.
- `model.py` — `NFW_Delta_Sigma(M200m, C200m, Z, FMISS, SIGMA_OFF, BCG_B_MASS, radii)`, vectorised (N clusters × M radii); loads `misc_NFW_*.fits`, `W_johnston.fits`, `test_matterpower.dat` at import.
- `triaxial.py` — exact elliptical-projection shear of a triaxial NFW; table `triaxial_nfw_shear_table.npz` built on first use (`XLENSING_CACHE`).
- `sims.py` — random metacal-like catalogues, spherical and triaxial shear injection, shape sampling.
- `fitting.py` — Gaussian likelihood and prior factories.
- `testing.py` — legacy toy generator used by the example notebooks.
- `build_miscentering_table.py` — regenerates the miscentring table.

## Conventions

Angles in radians, distances in physical Mpc, masses in M☉ (no h), ΔΣ in M☉/pc². Background selection is the caller's job. Keep every public function documented (numpy-style docstring) and covered by `tests/`.
