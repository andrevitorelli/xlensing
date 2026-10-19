"""Regenerate the miscentred-NFW Delta Sigma lookup table (2026-09-28).

The table gives the dimensionless miscentred excess surface density
Delta Sigma_mis(x; xi) of an NFW halo, in units of 2 r_s delta_c rho_m (the
units of xlensing.model.SigX), as a function of x = R / r_s and xi =
sigma_off / r_s, for centre offsets with a Rayleigh distribution of scale
sigma_off.  A Rayleigh offset magnitude with uniform direction is a 2D
Gaussian offset of width sigma_off per axis, so the miscentred surface density
is the NFW surface density convolved with that Gaussian (e.g. Johnston et al.
2007):

    Sigma_mis(x) = int_0^inf r Sigma(r) / xi^2 exp(-(r^2 + x^2) / (2 xi^2)) I0(r x / xi^2) dr,
    Sigma_bar_mis(<x) = (2 / x^2) int_0^x Sigma_mis(r) r dr,
    Delta Sigma_mis(x) = Sigma_bar_mis(<x) - Sigma_mis(x).

Validated against an independent Hankel-transform calculation (the projected
NFW's analytic 2D Fourier transform times exp(-q^2 xi^2 / 2), integrated
against J2): agreement to <2e-6 in absolute terms (<0.2% relative) for
xi = 0.3-2.5.  The 2020 table this replaces is equally accurate inside its
grid (xi = 0.1-10, errors <2e-3 of the centred signal), but its spline
extrapolates badly below xi = 0.1 (errors up to ~70% of the centred signal at
xi = 0.02), which a free sigma_off can reach; this table extends the grid to
xi = 1e-3, and xlensing.model clamps xi to the tabulated range.

usage: python -m xlensing.build_miscentering_table   (writes the two FITS files next to model.py)
"""
from __future__ import annotations

import os

import numpy as np
from astropy.table import Table
from scipy.special import i0e

from xlensing.model import SigX

HERE = os.path.dirname(__file__)
X_TAB = np.logspace(-3, 3, 1000)       # x = R / r_s at which the table is stored
XI_TAB = np.logspace(-3, np.log10(20.0), 100)   # xi = sigma_off / r_s (2020 table: 0.1-10, 20 values)


def _sigma(x):
    """SigX for arrays of any shape, with x floored at 1e-12."""
    x = np.maximum(np.asarray(x, dtype=float), 1e-12)
    return SigX(x.ravel()).reshape(x.shape)


def sigma_mis(x, xi, n=256, width=10.0):
    """Miscentred NFW surface density at radii x (array), for Rayleigh scale xi."""
    t, w = np.polynomial.legendre.leggauss(n)
    x = np.asarray(x, dtype=float)
    lo = np.maximum(0.0, x - width * xi)
    hi = x + width * xi
    r = lo[:, None] + (hi - lo)[:, None] * 0.5 * (t[None, :] + 1.0)      # (Nx, n)
    wr = (hi - lo)[:, None] * 0.5 * w[None, :]
    # exp(-(r^2+x^2)/2xi^2) I0(r x/xi^2) = exp(-(r-x)^2/2xi^2) i0e(r x/xi^2)
    kern = r / xi**2 * np.exp(-(r - x[:, None]) ** 2 / (2 * xi**2)) * i0e(r * x[:, None] / xi**2)
    return np.sum(wr * kern * _sigma(r), axis=1)


def delta_sigma_mis(x_out, xi, n_fine=6000):
    """Dimensionless miscentred Delta Sigma at x_out for Rayleigh scale xi."""
    xf = np.logspace(-6, np.log10(x_out.max()) + 0.01, n_fine)
    s = sigma_mis(xf, xi)
    s0 = sigma_mis(np.array([0.0]), xi)[0]                     # finite central value
    # cumulative int_0^x Sigma(r) r dr on the log grid (r dr = r^2 dln r), plus the core below xf[0]
    lnx = np.log(xf)
    integrand = s * xf**2
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(lnx))])
    cum += 0.5 * s0 * xf[0] ** 2
    sbar = 2.0 * cum / xf**2
    return np.interp(np.log(x_out), lnx, sbar - s)


def build(out_dir=HERE):
    """Compute the table on (X_TAB, XI_TAB) and write misc_NFW_radii.fits and misc_NFW_profiles.fits to ``out_dir``.

    Returns the (N_xi, N_x) array of dimensionless miscentred Delta Sigma.
    Takes a few minutes.
    """
    prof = np.array([delta_sigma_mis(X_TAB, xi) for xi in XI_TAB])          # (N_xi, N_x)
    Table({"col0": X_TAB}).write(os.path.join(out_dir, "misc_NFW_radii.fits"), overwrite=True)
    Table({"col0": XI_TAB, "col1": prof}).write(os.path.join(out_dir, "misc_NFW_profiles.fits"), overwrite=True)
    return prof


if __name__ == "__main__":
    build()
    print("wrote misc_NFW_radii.fits, misc_NFW_profiles.fits in", HERE)
