# xlensing user manual

xlensing measures, models and simulates the weak-lensing excess surface density
ΔΣ(R) around galaxy clusters. This manual walks through the three parts of the
package and gives the conventions you need to use them together. Every public
function has a docstring with its full parameter list: run `help(function)`.

Contents

1. Installation
2. Conventions
3. Measuring ΔΣ from a shear catalogue
4. Modelling ΔΣ
5. Simulating sheared catalogues
6. Fitting
7. Cosmology
8. Lookup tables and environment variables
9. Testing and benchmarks

---

## 1. Installation

```bash
git clone https://github.com/andrevitorelli/xlensing.git
cd xlensing
pip install .            # or `pip install -e .` for development
pip install ".[test]"    # adds pytest and pytest-cov
```

Requirements: Python ≥ 3.9, numpy, scipy, astropy, ngmix (simulations only),
and tqdm. The package data (the miscentring and two-halo lookup tables and a
linear power spectrum) are installed with the package.

## 2. Conventions

| Quantity | Unit |
|---|---|
| Angles (RA, Dec, separations) | radians |
| Distances and radii | physical Mpc |
| Masses | M☉ (no factors of h) |
| ΔΣ, Σ | M☉ pc⁻² (the model and the estimators divide M☉ Mpc⁻² by 10¹²) |
| Σ_crit inside the estimators | M☉ pc⁻² |

* **Halo mass definition.** Masses are M₂₀₀ₘ, defined with
  `cosmo.cluster_overdensity` (200) times `cosmo.rhoM(z)`. Note that
  `rhoM(z)` is Ω_m,0 ρ_crit(z), not Ω_m,0 (1+z)³ ρ_crit,0; see its
  docstring.
* **Background selection is the caller's job.** The measurement functions
  apply no redshift cut. Pass only the sources you consider background.
* **Ellipticity sign convention.** The sign convention follows the input
  catalogue (RA = −e₁). The `sys_angle` argument rotates the polar angle if
  your catalogue uses another convention.

## 3. Measuring ΔΣ from a shear catalogue (`xlensing.data`)

Measuring a profile takes two steps.

**Step 1: per-galaxy quantities around one cluster.**

```python
from xlensing import data
cluster = (ra_c, dec_c, z_c)                          # radians, radians, redshift
sources = (ra, dec, z, e1, e2, w, R11, R12, R21, R22)  # tuple / (10, N) array of per-galaxy columns
res = data.metacal_cluster_lensing(cluster, sources, radius=5.0)   # sources within 5 Mpc
```

`res` is a dict of per-galaxy arrays:
* `'Critical Density'` (Σ_crit);
* `'Tangential Shear'` and `'Cross Shear'` (e_t, e_×);
* `'Radial Distance'` (R in Mpc);
* `'Polar Angle'`;
* `'Weights'`;
* `'Mult. Bias'` (R_T − 1, the tangential response minus one);
* `'Count'`.

For lensfit-style catalogues use `data.lensfit_cluster_lensing`, which
takes a multiplicative bias m instead of a response matrix.

**Step 2: binned profiles.** Stack the per-galaxy quantities into a "stake",
a (6, N) array with rows Σ_crit, e_t, e_×, w, R, M:

```python
import numpy as np
stake = np.vstack([res['Critical Density'], res['Tangential Shear'], res['Cross Shear'],
                   res['Weights'], res['Radial Distance'], res['Mult. Bias']])
edges = np.logspace(-0.9, 0.7, 17)
bins = np.column_stack([edges[:-1], edges[1:]])       # (Nbins, 2) radial bin edges [Mpc]
```

Then choose one of the estimators:

| Function | What it returns |
|---|---|
| `data.signal(stake, bins)` | ΔΣ_t and ΔΣ_× of one cluster (no errors). |
| `data.bootstrap_signal(stake, bins, Nboot=200, sigma_e=None, floor_rmax=None)` | Single-cluster ΔΣ and covariance from a bootstrap over galaxies. Bins with too few galaxies are returned as NaN. `sigma_e` / `floor_rmax` floor the variance of sparse bins at the shape-noise value. |
| `data.stacked_signal([stake1, stake2, ...], bins, Nboot=200)` | Stacked, boost-corrected ΔΣ with a covariance from a bootstrap over clusters. Returns `(sigmas, boosts, boosts_cov, sigmas_cov, xigmas, xigmas_cov)`. The boost factor B(R) is the excess of source counts over a random catalogue. |
| `data.single_cluster([stake], bins, Nboot=500)` | The same as `stacked_signal`, with a galaxy bootstrap, for one cluster. |

The ΔΣ estimator is Σ_j e_t,j w_j / Σ_crit,j divided by
Σ_j R_j w_j / Σ_crit,j², per bin.

Helpers: `data.sigmacrit(z_l, z_s)`, `data.equatorial_to_polar`,
`data.tangential_response`, `data.cap_area`, `data.annular_area`.

## 4. Modelling ΔΣ (`xlensing.model`)

```python
from xlensing import model
out = model.NFW_Delta_Sigma(M200m, C200m, Z, FMISS, SIGMA_OFF, BCG_B_MASS, radii)
```

All arguments except `radii` are (N,) arrays, one entry per cluster, and
`radii` is an (M,) array in Mpc. Every output is an (N, M) array in M☉ pc⁻²:

| Key | Term |
|---|---|
| `'NFW Signal'` | centred NFW (analytic, Wright & Brainerd 2000) |
| `'Miscentered Signal'` | NFW averaged over Rayleigh-distributed centre offsets of scale `SIGMA_OFF` (lookup table) |
| `'BCG Signal'` | point mass `BCG_B_MASS` |
| `'Two-halo term'` | Johnston et al. (2007) two-halo term with the Tinker et al. (2010) bias |
| `'Signal'` | BCG + (1 − FMISS) NFW + FMISS Miscentered + two-halo |

The whole computation is vectorised, so evaluating tens of thousands of
clusters per call is the intended use (see the benchmarks in the README).

Lower-level pieces:
* NFW: `SigX`, `SigBar`, `NFW_delta_c`, `r_vir_m`, `r_vir_c`,
  `Delta_Sigma_NFW_off_x`.
* Two-halo term: `Delta_Sigma_l`, `Bias`, `Tinker_bias`, `peak_nu`,
  `sigma_squared`, `D`.

## 5. Simulating sheared catalogues (`xlensing.sims`, `xlensing.triaxial`)

* `sims.make_simple_random_cat(density, width_rad, zrange, shape_noise, seed)`
  makes a random metacalibration-like catalogue, with uniform positions and
  ngmix shape noise.
* `sims.apply_NFW_shear_region(cluster, galaxies)` applies the shear of a
  spherical NFW halo `[RA, DEC, z, M200, C200]` to a catalogue with
  columns RA, DEC, z, e1, e2, using exact shear composition.
* `sims.apply_triaxial_NFW_shear_region(cluster, galaxies, response=None, model="exact")`
  applies the shear of a triaxial NFW halo
  `[RA, DEC, z, M200, C200, a, b, c, theta_los, phi_los(, pa)]`.
  * With `model="exact"`, the projected surface density is exactly
    elliptical, includes the line-of-sight amplitude, and its shear comes
    from the Schramm (1990) / Keeton (2001) integrals in
    `xlensing.triaxial` (tabulated once and interpolated per galaxy).
  * Pass `response=` to scale the injected shear by a per-galaxy
    metacalibration response.
* `sims.sample_triaxial_shapes(z, M, rng=...)` draws random axis ratios and
  isotropic lines of sight.
* `sims.project_ellipsoid(a, b, c, theta_los, phi_los)` gives the projected
  semi-axes and position angle.

The first call to the exact triaxial shear builds a table
(`triaxial_nfw_shear_table.npz`, about 1 minute). See §8 for where it is
stored.

## 6. Fitting (`xlensing.fitting`)

Three factories return log-likelihood and log-prior functions of a parameter
vector θ, for use with any sampler (emcee, nautilus, ...):

```python
from functools import partial
from xlensing import fitting, model

def nfw_model(theta, radii, z):
    logM, c = theta
    return model.NFW_Delta_Sigma(np.array([10**logM]), np.array([c]), np.array([z]),
                                 np.zeros(1), np.full(1, 0.3), np.zeros(1), radii)['Signal'][0]

like = fitting.ln_gaussian_likelihood_maker((dsigma, cov), partial(nfw_model, radii=r, z=0.3))
priors = [fitting.ln_flat_prior_maker(13, 16, 0), fitting.ln_gaussian_prior_maker(5, 2, 1)]
```

For a covariance estimated from n realisations, pass
`hartlap=(n - p - 2)/(n - 1)`.

## 7. Cosmology (`xlensing.cosmo`)

Planck 2018 is the default. The module exposes:
* functions: `DA(z1, z2)`, `H(a)`, `rhocz(z)`, `rhoM(z)`, `scale_factor(z)`;
* constants: `lightspeed`, `gravity`, `cluster_overdensity`,
  `collapse_density`.

To change the cosmology, reassign `cosmo.cosmology`, `cosmo.h`,
`cosmo.OmegaM` and `cosmo.OmegaK` before importing `xlensing.model`, which
tabulates the growth factor and σ(R) at import time.

## 8. Lookup tables and environment variables

| File | Content |
|---|---|
| `misc_NFW_radii.fits`, `misc_NFW_profiles.fits` | miscentred NFW ΔΣ on x = R/r_s ∈ [10⁻³, 10³], ξ = σ_off/r_s ∈ [10⁻³, 20]; regenerate with `python -m xlensing.build_miscentering_table` |
| `*_legacy2020.fits` | the original 2020 table (ξ ∈ [0.1, 10]) |
| `W_johnston.fits` | projected linear correlation function for the two-halo term |
| `test_matterpower.dat` | z = 0 linear power spectrum (σ₈ = 0.802) |
| `triaxial_nfw_shear_table.npz` | built on first use of the exact triaxial shear |

| Variable | Effect |
|---|---|
| `XLENSING_MISCENTERING_TABLE=legacy2020` | use the 2020 miscentring table (to reproduce older fits) |
| `XLENSING_CACHE=/path` | directory for the triaxial shear table. Default: the package directory if it is writable, otherwise `~/.cache/xlensing`. |

## 9. Testing and benchmarks

```bash
pytest                                   # the test suite (about 130 tests, a few seconds)
pytest --cov=xlensing --cov-report=term  # with coverage
python scripts/benchmark.py              # timings quoted in the README
```
