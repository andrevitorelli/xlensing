# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:50:50 2015

@author: andrezvitorelli
"""

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
    return (cosmo.cluster_overdensity/3.)*((conc**3)/(np.log(1.+conc)-conc/(1.+conc)))

def r_vir_c(z,M):
    dc = cosmo.cluster_overdensity #\Delta (standard = 200)
    result = ((3.*M)/(cosmo.rhocz(z)*4.*np.pi*dc))**(1./3.)
    return result

def r_vir_m(z,M):
    dc = cosmo.cluster_overdensity #\Delta (standard = 200)
    result = ((3.*M)/(cosmo.rhoM(z)*4.*np.pi*dc))**(1./3.)
    return result

def SigX(x):
    result = np.zeros_like(x)
    lesser = x<1
    equal = x==1
    greater = x>1
    result[equal]=1/3
    result[lesser] =(1 - 2/np.sqrt(1-x[lesser]**2)*np.arctanh(np.sqrt((1-x[lesser])/(1+x[lesser]))))/(x[lesser]**2-1)
    result[greater]=(1 - 2/np.sqrt(x[greater] **2-1)*np.arctan(np.sqrt((x[greater]-1)/(1+x[greater]))))/(x[greater]**2-1)
    return result
def SigBar(x):
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
MscRadii = Table.read(resources+"/misc_NFW_radii.fits")
Profs = Table.read(resources+"/misc_NFW_profiles.fits")
print("Lookup tables loaded!")
spline = BSpline(MscRadii['col0'],Profs['col0'],Profs['col1'].T,s=0)

def Delta_Sigma_NFW_off_x(matrix, vector):
    """Vectorized evaluation for (N×M matrix, N vector) pairs"""
    N, M = matrix.shape
    x_flat = matrix.ravel()
    y_flat = np.repeat(vector, M)
    z_flat = spline.ev(x_flat, y_flat)
    return z_flat.reshape(N, M)




#read and interpolate the power spectrum
matter_power_camb = ascii.read(resources+"/test_matterpower.dat")
k, Pk  = matter_power_camb['col1'], matter_power_camb['col2']
PowerSpec = USpline(k,Pk,s=0,k=5,ext=1)

#read and interpolate matter correlation
# matt_corr_table = Table.read(resources+"/matt_corr.fits")
# raios_matt, matt_cor = matt_corr_table['col0'], matt_corr_table['col1']
# matter_correlation_norm = USpline(raios_matt,
#                                   np.array(matt_cor)/0.6434659,
#                                   s=0,k=5, ext=1)#0.802**2

#Window function
Window = lambda x: (3/(x*x*x))*(np.sin(x)-x*np.cos(x))

def D1(a_array, N=1000):
    """Simplest vectorized version"""
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
    return D(cosmo.scale_factor(z))**2 * _sigma2_R_spline(R)

#sigma_8
def sigma8():
    return np.sqrt(sigma_squared(0.,8.))

#Peak height (delta_c/sigma)
def peak_nu (z, Mvir):
    radius = (3*Mvir/(12.56637*cosmo.rhoM(z)))**(1/3) #4*pi = 12.56637...
    niu = cosmo.collapse_density/np.sqrt(sigma_squared(z,radius))
    return niu

#Peak Bias
def Tinker_bias(nu):
    y = np.log10(cosmo.cluster_overdensity)
    """Tinker et al 2010:"""
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
    """
    Two-halo Delta_Sigma for 1D arrays z (N,) and R (M,).
    Returns array of shape (N, M).
    """
    u = np.outer(1 + z, R)  # (N, M)
    G = _G_spline(u.ravel()).reshape(u.shape)
    return 2 / R**2 * G - Sigma_l(z[:, None], R[None, :])

def NFW_Delta_Sigma(M200m, C200m, Z, FMISS, SIGMA_OFF, BCG_B_MASS, radii):
    """
    NFW MODEL:

    - M200m: array of masses in Msun
    - C200m: array of concentrations
    - Z: redshifts
    - FMISS: array of fraction of miscentered signal
    - SIGMA_OFF: array of characteristic miscentering
    - BCG_B_MASS: array of baryonic masses of central galaxies
    - radii: a collection of radii at which we calculate the model (mpc)

    RETURNS:

    A dict with each value an (N, M) array giving the signal in Msun/pc2.
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
