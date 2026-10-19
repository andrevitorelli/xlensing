# xlensing

**Fast weak-lensing measurement, modelling and simulation of galaxy-cluster ΔΣ profiles.**

xlensing measures the excess surface density ΔΣ(R) around galaxy clusters from
shear catalogues, models it, and simulates sheared catalogues to test the
whole chain. It was written for fitting tens of thousands of individual
cluster profiles at once, so every model evaluation is vectorised over
clusters. It is the lensing engine behind the joint per-cluster fits of
[destacking](https://github.com/andrevitorelli/destacking).

## Speed

Measured with `python scripts/benchmark.py` on a single core of an Intel Core
Ultra 7 255H laptop CPU (wall clock, median of repeated runs):

| Operation | Size | Time |
|---|---|---|
| Full ΔΣ model: centred + miscentred NFW + two-halo term | 1,000 clusters × 16 radii | 8 ms |
| | 100,000 clusters × 16 radii | 0.9 s (**9 µs per cluster**) |
| Per-galaxy lensing quantities (Σ_crit, e_t, e_×, response) | 100,000 sources | 0.16 s (1.6 µs per source) |
| Binned ΔΣ + 200-resample bootstrap covariance | 22,000 sources, 16 bins | 0.25 s |
| Exact triaxial NFW shear injection | 40,000 sources | 0.3 s (8 µs per source) |

A full nine-parameter likelihood over 16,000 clusters therefore costs about
0.15 s, which makes nested sampling of joint per-cluster fits practical.

The speed comes from three design choices:
* **Analytic NFW terms.** The centred profile is closed-form.
* **Lookup tables.** The miscentred NFW profile and the two-halo term are
  pre-computed tables evaluated with 2D splines, not integrals at run time.
* **No Python loops over clusters.** Every model term is an array operation
  over (clusters × radii).

## What it provides

**Measurement** (`xlensing.data`)
* Tangential and cross shear, Σ_crit and weights around a cluster, for
  metacalibration catalogues (2×2 response matrix) or lensfit-style
  catalogues (multiplicative bias).
* Response-weighted ΔΣ estimator per radial bin.
* Single-cluster profiles with a galaxy-bootstrap covariance, and an
  optional shape-noise variance floor for sparse bins.
* Stacked profiles with a cluster-bootstrap covariance that propagates the
  boost-factor correction jointly with the signal.

**Model** (`xlensing.model`)
* NFW ΔΣ (Wright & Brainerd 2000) with masses M₂₀₀ₘ and concentrations c₂₀₀ₘ.
* Miscentring: Rayleigh-distributed centre offsets, from a lookup table
  valid down to σ_off/r_s = 10⁻³.
* Two-halo term (Johnston et al. 2007) with the Tinker et al. (2010) bias.
* Central-galaxy point mass.

**Simulation** (`xlensing.sims`, `xlensing.triaxial`)
* Random metacalibration-like source catalogues with ngmix shape noise.
* Spherical NFW shear injection.
* **Exact triaxial NFW lensing.** The projected surface density is exactly
  elliptical, including the line-of-sight amplitude, and the shear comes
  from the Schramm (1990) / Keeton (2001) integrals, tabulated once.
  Optional per-galaxy metacalibration response scaling.
* Random triaxial shapes and isotropic orientations.

**Fitting helpers** (`xlensing.fitting`)
* Gaussian log-likelihood (with an optional Hartlap factor) and flat or
  Gaussian log-priors, for any sampler.

## Installation

```bash
pip install git+https://github.com/andrevitorelli/xlensing
# or, for development:
git clone https://github.com/andrevitorelli/xlensing && cd xlensing && pip install -e ".[test]"
```

Requirements: Python ≥ 3.9, numpy, scipy, astropy, ngmix, tqdm.

## Quick start

```python
import numpy as np
from xlensing import model

radii = np.logspace(-1, 1, 20)                       # Mpc
out = model.NFW_Delta_Sigma(
    M200m=np.array([3e14, 1e15]),                    # Msun
    C200m=np.array([5.0, 4.0]),
    Z=np.array([0.3, 0.5]),
    FMISS=np.array([0.2, 0.1]),                      # miscentred fraction
    SIGMA_OFF=np.array([0.3, 0.2]),                  # Rayleigh offset scale, Mpc
    BCG_B_MASS=np.zeros(2),                          # central point mass, Msun
    radii=radii,
)
out["Signal"]        # (2, 20) total Delta Sigma in Msun/pc^2
```

Measuring a profile from a catalogue, simulating a sheared catalogue and
fitting are covered step by step in the **[user manual](docs/MANUAL.md)**.
Every function has a docstring (`help(xlensing.data.stacked_signal)`).

## Tests

```bash
pytest                                     # about 130 tests, a few seconds
pytest --cov=xlensing --cov-report=term    # 94% line coverage
```

## Citing

If you use xlensing, please cite Vitorelli & Cypriano (2026), *Joint fit of
galaxy cluster scaling relations from individual weak lensing data*, and
Vitorelli et al. (2018).

## License

MIT (see `LICENSE`).
