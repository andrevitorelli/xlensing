# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

`xlensing` is a Python package for cross-correlation weak gravitational lensing of galaxy cluster systems. It computes the excess surface mass density (ΔΣ) profile around galaxy clusters from shear catalogs, models it with NFW profiles, and fits cosmological parameters.

## Installation

```bash
pip install -e .
```

The package depends on `numpy`, `scipy`, `astropy`, `ngmix` (for sims), and `tqdm`.

## Module Architecture

The six modules form a pipeline from raw catalogs to fitted parameters:

- **`cosmo.py`** — Global cosmological constants and functions (Planck18 defaults). Provides `DA(z1,z2)`, `rhocz(z)`, `rhoM(z)`. Cosmology can be swapped by reassigning `cosmo.cosmology`.

- **`data.py`** — Core measurement layer. Takes cluster + source galaxy tuples and returns per-galaxy lensing quantities (`sigmacrit`, tangential/cross shear, radial distance, weights). Two entry points:
  - `lensfit_cluster_lensing()` — for lensfit-format catalogs (ellipticity + multiplicative bias M)
  - `metacal_cluster_lensing()` — for metacalibration catalogs (ellipticity + response matrix R11/R12/R21/R22)
  
  Both return a dict of arrays that must be stacked into a `(6, N)` ndarray (rows: Σ_crit, e_t, e_x, W, R, M) before being passed to `stacked_signal` or `single_cluster`.

- **`model.py`** — Physical model for ΔΣ. The main function `NFW_Delta_Sigma(M200m, C200m, Z, FMISS, SIGMA_OFF, BCG_B_MASS, radii)` returns a dict with `'Signal'` and component terms. Accepts arrays for batch evaluation (vectorized over cluster parameters). Loads three FITS lookup tables at import time (`misc_NFW_radii.fits`, `misc_NFW_profiles.fits`, `W_johnston.fits`) — these must remain in the package directory.

- **`fitting.py`** — Likelihood and prior factories for MCMC. `ln_gaussian_likelihood_maker(data, model)` returns a closure; combine with priors from `ln_flat_prior_maker` / `ln_gaussian_prior_maker` for use with `emcee`.

- **`sims.py`** — Simulation utilities. `make_simple_random_cat()` generates a metacal-format galaxy catalog; `apply_NFW_shear_region()` injects an NFW shear signal. Uses `ngmix` for realistic shape noise.

- **`utils.py`** — Currently empty placeholder.

## Key Data Conventions

- All angles are in **radians**.
- Masses are in **M☉**, radii/distances in **Mpc**.
- ΔΣ is returned in **M☉/pc²** (divided by 1e12 internally for numerical stability).
- Background galaxy selection cuts are hardcoded in `data.py`: `z_source > 1.1 * z_cluster + 0.1` (lensfit) or `+ 0.2` (metacal).
- The stacked signal array layout expected by `signal()` / `stacked_signal()` / `single_cluster()` is row-indexed: `[0]` Σ_crit, `[1]` e_t, `[2]` e_x, `[3]` W, `[4]` R, `[5]` M.

## Notebooks

`notebooks/` contains worked examples:
- `xlensing_example.ipynb` — end-to-end measurement example with a mock catalog
- `Xlensing_Sims.ipynb` — simulation injection and recovery
- `better_parallel_clusters.ipynb` — parallelised multi-cluster stacking

`notebooks/mock_galaxy_catalog.fits` and `mock_results.pickle` are pre-generated test data used by the example notebooks.
