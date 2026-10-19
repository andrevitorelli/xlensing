"""Cosmology and physical constants used throughout xlensing.

The default cosmology is Planck 2018 (``astropy.cosmology.Planck18``).  To use
another one, reassign the module attributes before computing anything, e.g.::

    from astropy.cosmology import FlatLambdaCDM
    import xlensing.cosmo as cosmo
    cosmo.cosmology = FlatLambdaCDM(H0=70, Om0=0.3)
    cosmo.h, cosmo.OmegaM, cosmo.OmegaK = 0.7, 0.3, 0.0

Note that :mod:`xlensing.model` pre-computes some tables (growth factor,
sigma(R)) at import time with the cosmology in force at that moment.

Units: distances in Mpc, masses in Msun, densities in Msun/Mpc^3, velocities
in km/s.
"""
from astropy.cosmology import FlatLambdaCDM, Planck18  # noqa: F401  (FlatLambdaCDM kept for users swapping cosmology)
from numpy import sqrt


# Constants
lightspeed = 299792.458   #: speed of light [km/s]
cluster_overdensity = 200  #: halo overdensity Delta (masses are M_Delta)
collapse_density = 1.686   #: linear collapse threshold delta_c
gravity = 4.302e-9         #: Newton's G [Mpc (km/s)^2 / Msun]
h = Planck18.h             #: dimensionless Hubble parameter

# Default cosmological parameters
OmegaM = Planck18.Om0      #: matter density parameter today
OmegaK = Planck18.Ok0      #: curvature density parameter today
cosmology = Planck18       #: astropy cosmology used for distances


def DA(z1, z2):
    """Angular diameter distance [Mpc] between redshifts ``z1`` and ``z2`` (``z2 > z1``).

    ``DA(0, z)`` is the angular diameter distance to ``z``.  Accepts scalars
    or broadcastable arrays.
    """
    return cosmology.angular_diameter_distance_z1z2(z1, z2).value


def H(a):
    """Hubble rate H(a) [km/s/Mpc] at scale factor ``a`` for (OmegaM, OmegaK, Lambda = 1 - OmegaM)."""
    return 100. * h * sqrt(OmegaM * (a) ** (-3) + OmegaK * (a) ** (-2) + 1. - OmegaM)


def scale_factor(z):
    """Scale factor a = 1 / (1 + z) for redshift(s) ``z``."""
    return 1. / (1. + z)


def rhocz(z):
    """Critical density of the Universe at redshift ``z`` [Msun/Mpc^3], 3 H^2 / (8 pi G)."""
    return 27746677.666 * (H(scale_factor(z)) ** 2)  # 3/(8 pi G) in Msun, km/s, Mpc


def rhoM(z):
    """Matter density used for the M_200m definition [Msun/Mpc^3]: OmegaM * rho_crit(z).

    Note: this uses the present-day OmegaM times the critical density at
    ``z``.  It equals the mean matter density Omega_m (1+z)^3 rho_crit,0
    only at z = 0; at z > 0 the two differ by E(z)^2 / (1+z)^3.  It is kept
    unchanged for consistency with published results.
    """
    return OmegaM * rhocz(z)
