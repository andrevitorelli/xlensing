# -*- coding: utf-8 -*-
"""Excess surface density (Delta Sigma) model of galaxy clusters.

The model is the sum of four terms, all evaluated for N clusters at M radii at
once and returned as (N, M) arrays in Msun/pc^2:

* a centred NFW halo (analytic, Wright & Brainerd 2000);
* a miscentred NFW halo, for centre offsets with a Rayleigh distribution of
  scale ``sigma_off`` (pre-computed lookup table, see
  :mod:`xlensing.build_miscentering_table`);
* a point mass for the baryonic mass of the central galaxy;
* the two-halo term of Johnston et al. (2007), with the Tinker et al. (2010)
  halo bias.

Masses are M_200m (200 times the mean matter density), in Msun without h;
radii are physical Mpc.  The lookup tables and the linear power spectrum are
read once at import time.  The cosmology is that of :mod:`xlensing.cosmo`.

Main entry point: :func:`NFW_Delta_Sigma`.
"""
import os

########################################Model Basics############################
#Radius as a function of mass for a given mass density at redshift z
import numpy as np
from xlensing import cosmo
from astropy.table import Table
from scipy.interpolate import UnivariateSpline as USpline
from scipy.interpolate import RectBivariateSpline as BSpline
from astropy.io import fits, ascii
from scipy import integrate

from pathlib import Path
#resources = str(Path(__file__).parent)
def NFW_delta_c(conc):
    """NFW characteristic overdensity delta_c for concentration ``conc``.

    delta_c = (Delta/3) c^3 / [ln(1+c) - c/(1+c)], with Delta =
    ``cosmo.cluster_overdensity`` (200).  Works on scalars and arrays.
    """
    return (cosmo.cluster_overdensity/3.)*((conc**3)/(np.log(1.+conc)-conc/(1.+conc)))

def r_vir_c(z,M):
    """Radius [Mpc] enclosing mass ``M`` [Msun] at Delta times the *critical* density at redshift ``z``."""
    dc = cosmo.cluster_overdensity #\Delta (standard = 200)
    result = ((3.*M)/(cosmo.rhocz(z)*4.*np.pi*dc))**(1./3.)
    return result

def r_vir_m(z,M):
    """Radius [Mpc] enclosing mass ``M`` [Msun] at Delta times the *mean* matter density at redshift ``z`` (r_200m)."""
    dc = cosmo.cluster_overdensity #\Delta (standard = 200)
    result = ((3.*M)/(cosmo.rhoM(z)*4.*np.pi*dc))**(1./3.)
    return result

def SigX(x):
    """Dimensionless projected NFW surface density at x = R / r_s.

    Multiply by 2 r_s delta_c rho_m to obtain Sigma(R).  ``x`` must be a
    numpy array (any shape); x = 1 is handled exactly.
    """
    result = np.zeros_like(x)
    lesser = x<1
    equal = x==1
    greater = x>1
    result[equal]=1/3
    result[lesser] =(1 - 2/np.sqrt(1-x[lesser]**2)*np.arctanh(np.sqrt((1-x[lesser])/(1+x[lesser]))))/(x[lesser]**2-1)
    result[greater]=(1 - 2/np.sqrt(x[greater] **2-1)*np.arctan(np.sqrt((x[greater]-1)/(1+x[greater]))))/(x[greater]**2-1)
    return result
def SigBar(x):
    """Dimensionless mean NFW surface density inside x = R / r_s.

    Same units as :func:`SigX`; Delta Sigma(x) is proportional to
    ``SigBar(x) - SigX(x)``.
    """
    result = np.zeros_like(x)
    lesser = x<1
    equal = x==1
    greater = x>1
    result[equal]  =2.*(1+np.log(0.5))
    result[lesser] = 2/x[lesser]**2 * ( 2./np.sqrt(1-x[lesser]**2)*np.arctanh(np.sqrt((1-x[lesser])/(1+x[lesser]))) + np.log(x[lesser]/2) )
    result[greater]= 2/x[greater]**2 * ( 2./np.sqrt(x[greater]**2-1)*np.arctan(np.sqrt((x[greater]-1)/(1+x[greater]))) + np.log(x[greater]/2) )
    return result

#load NFW miscentered profile
resources = str(Path(__file__).parent)
# Miscentred-NFW Delta Sigma table (build: xlensing/build_miscentering_table.py).
# Default: the 2026-09-28 table (xi = sigma_off/r_s from 1e-3 to 20).  Set
# XLENSING_MISCENTERING_TABLE=legacy2020 to load the original 2020 table (xi =
# 0.1-10), e.g. to reproduce older fits.
_MISC_SUFFIX = "_legacy2020" if os.environ.get("XLENSING_MISCENTERING_TABLE", "") == "legacy2020" else ""
MscRadii = Table.read(resources+f"/misc_NFW_radii{_MISC_SUFFIX}.fits")
Profs = Table.read(resources+f"/misc_NFW_profiles{_MISC_SUFFIX}.fits")
spline = BSpline(MscRadii['col0'],Profs['col0'],Profs['col1'].T,s=0)
_XI_MIN, _XI_MAX = float(np.min(Profs['col0'])), float(np.max(Profs['col0']))

def Delta_Sigma_NFW_off_x(matrix, vector):
    """Dimensionless miscentred NFW Delta Sigma from the lookup table.

    ``matrix`` is an (N, M) array of x = R / r_s and ``vector`` an (N,)
    array of xi = sigma_off / r_s, one per cluster; returns an (N, M) array in
    the units of :func:`SigX`.

    xi (``vector``, sigma_off / r_s) is clamped to the tabulated range: the
    spline must not extrapolate (below the 2020 table's xi = 0.1 it was off by
    up to ~70% of the centred signal).  Below the new table's xi = 1e-3 the
    miscentred profile is indistinguishable from the centred one at any radius
    the pipeline uses."""
    N, M = matrix.shape
    x_flat = matrix.ravel()
    y_flat = np.repeat(np.clip(vector, _XI_MIN, _XI_MAX), M)
    z_flat = spline.ev(x_flat, y_flat)
    return z_flat.reshape(N, M)




#read and interpolate the power spectrum
matter_power_camb = ascii.read(resources+"/test_matterpower.dat")
k, Pk  = matter_power_camb['col1'], matter_power_camb['col2']
PowerSpec = USpline(k,Pk,s=0,k=5,ext=1)

#Window function
Window = lambda x: (3/(x*x*x))*(np.sin(x)-x*np.cos(x))

def D1(a_array, N=1000):
    """Unnormalised linear growth factor at scale factors ``a_array``.

    Heath (1977) integral D1(a) = H(a) int_0^a da' / (a' H(a'))^3 (times
    (100 h)^3), evaluated by the trapezoidal rule with ``N`` points.  Use
    :func:`D` for the growth factor normalised to D(a=1) = 1.
    """
    a_array = np.asarray(a_array)

    # Create 2D grid: rows = different a values, columns = integration points
    x_grid = np.outer(a_array, np.linspace(0, 1, N))
    x_grid = np.maximum(x_grid, 1e-10)  # avoid x=0

    # Vectorized computation
    H_x = cosmo.H(x_grid)
    integrand = (100 * cosmo.h / (x_grid * H_x))**3
    integrals = np.trapezoid(integrand, x_grid, axis=1)
    return cosmo.H(a_array) * integrals

dnorm = D1(np.array([1.]))[0]

# Precompute D(a) as a spline — avoids an N×1000 quadrature grid on every call.
# D1 is called once here at import; D(a) is then just N spline evaluations.
_a_D_grid = np.linspace(1e-3, 1.0, 500)
_D_spline = USpline(_a_D_grid, D1(_a_D_grid) / dnorm, s=0, k=3, ext=0)

def D(a):
    """Linear growth factor normalised to 1 today, at scale factor(s) ``a`` (spline of :func:`D1`)."""
    return _D_spline(np.asarray(a))

k_grid = np.logspace(-4, 4, 1000)  # Adjust as needed
k_squared = (k_grid**2) / 19.7392
power_spec = PowerSpec(k_grid)
k_grid = k_grid[:, None]

# Precompute sigma²(R) without the D(z) factor, which separates as sigma²(z,R)=D(z)²·sigma²_pure(R).
# Avoids rebuilding a 1000×N Window grid on every call; replaced by N spline evaluations.
_R_sigma_grid = np.logspace(-3, 3, 1000)
_sigma2_pure = np.trapezoid(
    k_squared[:, None] * power_spec[:, None] * Window(k_grid * _R_sigma_grid)**2,
    k_grid[:, 0], axis=0)
_sigma2_R_spline = USpline(_R_sigma_grid, _sigma2_pure, s=0, k=3, ext=0)

def sigma_squared(z, R):
    """Linear matter variance sigma^2(R, z) in a top-hat sphere of radius ``R`` [Mpc] at redshift ``z``.

    Computed from the tabulated z = 0 linear power spectrum
    (``test_matterpower.dat``, sigma_8 = 0.802) and scaled by D(z)^2.
    """
    return D(cosmo.scale_factor(z))**2 * _sigma2_R_spline(R)

#sigma_8
def sigma8():
    """sigma_8 of the tabulated z = 0 linear power spectrum (0.802)."""
    return np.sqrt(sigma_squared(0.,8.))

#Peak height (delta_c/sigma)
def peak_nu (z, Mvir):
    """Peak height nu = delta_c / sigma(M, z) of a halo of mass ``Mvir`` [Msun] at redshift ``z``.

    sigma is evaluated at the Lagrangian radius R = [3 M / (4 pi rho_m(z))]^(1/3).
    """
    radius = (3*Mvir/(12.56637*cosmo.rhoM(z)))**(1/3) #4*pi = 12.56637...
    niu = cosmo.collapse_density/np.sqrt(sigma_squared(z,radius))
    return niu

#Peak Bias
def Tinker_bias(nu):
    """Linear halo bias b(nu) of Tinker et al. (2010) for overdensity ``cosmo.cluster_overdensity``."""
    y = np.log10(cosmo.cluster_overdensity)
    A = 1.0 + 0.24*y*np.exp(-(4/y)**4)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
    c = 2.4
    result = 1 - A*(nu**a)/(nu**a+cosmo.collapse_density**a) +B*nu**b +C*nu**c

    return result

#sigma_8 squared
sig802 = sigma8()**2

#critical density at redshift zero
rhoc0 = cosmo.rhocz(0)

#Normalised bias
def Bias(z,Mvir):
    """Amplitude of the two-halo term, B(z, M) = rho_c0 b(nu) sigma_8^2 Omega_m D(z)^2.

    Multiplied by :func:`Delta_Sigma_l` it gives the two-halo Delta Sigma
    (Johnston et al. 2007).  ``z`` and ``Mvir`` are arrays of equal length.
    """
    B  = rhoc0*Tinker_bias(peak_nu(z,Mvir))*sig802*cosmo.OmegaM*D(cosmo.scale_factor(z))**2
    return B

#Johnston 2007 2ht
w_johnston_val = Table.read(resources+"/W_johnston.fits") #precalculated for fast interpolation function.
W_Johnston = USpline(w_johnston_val['col0'],w_johnston_val['col1'],s=0,k=5,ext=1)
Sigma_l = lambda z,R: (1+z)**2*W_Johnston((1+z)*R)

# Precompute G(u) = ∫₀ᵘ v·W_Johnston(v) dv.
# Substitution u=(1+z)x shows ∫₀ᴿ x·Σₗ(z,x)dx = G((1+z)R), collapsing the
# N×M×N_int integration array in Delta_Sigma_l to N×M spline evaluations.
_u_G_grid = np.logspace(-6, 4, 10000)
_G_vals = integrate.cumulative_trapezoid(_u_G_grid * W_Johnston(_u_G_grid), _u_G_grid, initial=0)
_G_spline = USpline(_u_G_grid, _G_vals, s=0, k=3, ext=3)  # ext=3: clamp to boundary outside range

def Delta_Sigma_l(z, R):
    """Projected linear matter correlation term of the two-halo Delta Sigma.

    For 1D arrays ``z`` (N,) and ``R`` (M,) [physical Mpc], returns the (N, M)
    array Delta Sigma_l(z, R) = Sigma_bar_l(<R) - Sigma_l(R), with Sigma_l the
    Johnston et al. (2007) projection of the linear correlation function
    (pre-computed table ``W_johnston.fits``).  Multiply by :func:`Bias` for the
    two-halo signal.
    """
    u = np.outer(1 + z, R)  # (N, M)
    G = _G_spline(u.ravel()).reshape(u.shape)
    return 2 / R**2 * G - Sigma_l(z[:, None], R[None, :])

def NFW_Delta_Sigma(M200m, C200m, Z, FMISS, SIGMA_OFF, BCG_B_MASS, radii):
    """Delta Sigma model of N clusters at M radii.

    Parameters
    ----------
    M200m : (N,) array
        Halo masses M_200m [Msun].
    C200m : (N,) array
        Concentrations c_200m.
    Z : (N,) array
        Cluster redshifts.
    FMISS : (N,) array
        Fraction of miscentred clusters (weight of the miscentred NFW term).
    SIGMA_OFF : (N,) array
        Rayleigh scale of the centre offsets [Mpc].
    BCG_B_MASS : (N,) array
        Baryonic point mass of the central galaxy [Msun]; zeros to omit it.
    radii : (M,) array
        Projected radii [physical Mpc].

    Returns
    -------
    dict
        ``'Signal'`` (the total), ``'NFW Signal'`` (centred NFW),
        ``'Miscentered Signal'`` (miscentred NFW, before the FMISS weight),
        ``'BCG Signal'`` and ``'Two-halo term'``, each an (N, M) array in
        Msun/pc^2, and ``'radii'``.  Total = BCG + (1 - FMISS) NFW +
        FMISS Miscentered + Two-halo.

    Examples
    --------
    >>> import numpy as np
    >>> from xlensing import model
    >>> r = np.logspace(-1, 1, 10)
    >>> out = model.NFW_Delta_Sigma(np.array([3e14]), np.array([5.]), np.array([0.3]),
    ...                             np.array([0.2]), np.array([0.3]), np.array([0.]), r)
    >>> out['Signal'].shape
    (1, 10)
    """

    rs   = r_vir_m(Z, M200m) / C200m                    # (N,)
    fact = 2 * rs * NFW_delta_c(C200m) * cosmo.rhoM(Z)  # (N,)
    xi   = SIGMA_OFF / rs                                # (N,)

    # X[n, m] = radii[m] / rs[n]  — (N, M) layout avoids all downstream transposes
    X = radii[np.newaxis, :] / rs[:, np.newaxis]         # (N, M)

    sigx    = SigX(X)   # (N, M)
    sigbarx = SigBar(X) # (N, M)

    signal_BCG        = BCG_B_MASS[:, np.newaxis] / (np.pi * radii[np.newaxis, :]**2) / 1e12  # (N, M)
    signal_NFW_centre = fact[:, np.newaxis] * (sigbarx - sigx) / 1e12                         # (N, M)
    signal_NFW_miscc  = fact[:, np.newaxis] * Delta_Sigma_NFW_off_x(X, xi) / 1e12             # (N, M)
    signal_2ht        = Bias(Z, M200m)[:, np.newaxis] * Delta_Sigma_l(Z, radii) / 1e12        # (N, M)

    signal_total = (signal_BCG
                    + (1 - FMISS[:, np.newaxis]) * signal_NFW_centre
                    +      FMISS[:, np.newaxis]  * signal_NFW_miscc
                    + signal_2ht)                                                               # (N, M)

    return {
        'Signal':             signal_total,
        'BCG Signal':         signal_BCG,
        'NFW Signal':         signal_NFW_centre,
        'Miscentered Signal': signal_NFW_miscc,
        'Two-halo term':      signal_2ht,
        'radii':              radii,
    }
