# xlensing

A Python package for cross-correlation weak gravitational lensing of galaxy cluster systems. It computes the excess surface mass density (ΔΣ) profile around galaxy clusters from shear catalogs, models it with NFW profiles, and fits cosmological parameters.

## Installation

```bash
pip install -e .
```

Dependencies: `numpy`, `scipy`, `astropy`, `ngmix` (for simulations), `tqdm`.

## Quick start

```python
import numpy as np
from xlensing import model

# N clusters × M radii — all outputs are (N, M) arrays
radii  = np.logspace(-1, 1, 20)       # Mpc
result = model.NFW_Delta_Sigma(
    M200m      = np.array([3e14, 1e15]),   # M☉
    C200m      = np.array([5.0,  4.0]),
    Z          = np.array([0.3,  0.5]),
    FMISS      = np.array([0.2,  0.1]),
    SIGMA_OFF  = np.array([0.3,  0.2]),    # Mpc
    BCG_B_MASS = np.array([5e11, 8e11]),   # M☉
    radii      = radii,
)
# result['Signal']  — total ΔΣ in M☉/pc²
# result['NFW Signal'], result['BCG Signal'], result['Miscentered Signal'], result['Two-halo term']
```

## Package structure

The six modules form a pipeline from raw catalogs to fitted parameters.

| Module | Role |
|---|---|
| `cosmo.py` | Cosmological constants and functions (Planck18 defaults). Provides `DA(z1,z2)`, `rhocz(z)`, `rhoM(z)`. Swap cosmology by reassigning `cosmo.cosmology`. |
| `data.py` | Core measurement layer. Computes Σ_crit, tangential/cross shear, and weights per background galaxy. Entry points: `lensfit_cluster_lensing()` and `metacal_cluster_lensing()`. |
| `model.py` | Physical NFW model. `NFW_Delta_Sigma()` returns the centred NFW, miscentred NFW, BCG point mass, and two-halo contributions — all vectorised over N clusters at M radii. |
| `fitting.py` | Likelihood and prior factories for MCMC. Use `ln_gaussian_likelihood_maker()` with priors from `ln_flat_prior_maker` / `ln_gaussian_prior_maker` and `emcee`. |
| `sims.py` | Simulation utilities. `make_simple_random_cat()` generates a metacalibration-format catalog; `apply_NFW_shear_region()` injects an NFW shear signal. |
| `utils.py` | Placeholder for future utilities. |

## Measurement pipeline

`data.py` exposes two entry points depending on the shape-measurement method:

- **`lensfit_cluster_lensing(cluster, sources, radius)`** — sources carry ellipticity components and a multiplicative bias scalar M.
- **`metacal_cluster_lensing(cluster, sources, radius)`** — sources carry ellipticity components and a 2×2 response matrix (R11, R12, R21, R22).

Both return a dict of per-galaxy arrays. Stack them into a `(6, N)` ndarray (rows: Σ_crit, e_t, e_x, W, R, M) and pass to `stacked_signal()` or `single_cluster()` to get the radially binned ΔΣ profile with bootstrap covariance.

## Data conventions

- All angles in **radians**.
- Masses in **M☉**, radii and distances in **Mpc**.
- ΔΣ is in **M☉/pc²**.
- Background selection cut in `data.py`: `z_source > 1.1 × z_cluster + 0.1` (lensfit) or `+ 0.2` (metacal).

## Notebooks

| Notebook | Description |
|---|---|
| `xlensing_example.ipynb` | End-to-end measurement with a mock catalog |
| `Xlensing_Sims.ipynb` | NFW shear injection and signal recovery |
| `better_parallel_clusters.ipynb` | Parallelised multi-cluster stacking |
