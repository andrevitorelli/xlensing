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
resources = "/home/andre/github/xlensing/xlensing"
MscRadii = Table.read(resources+"/misc_NFW_radii.fits")
Profs = Table.read(resources+"/misc_NFW_profiles.fits")
print("Lookup tables loaded!")
spline = BSpline(MscRadii['col0'],Profs['col0'],Profs['col1'].T,s=0)

def Delta_Sigma_NFW_off_x(matrix, vector):
    """Vectorized evaluation for (N×M matrix, N vector) pairs"""
    N, M = matrix.shape
    
    # Flatten matrix: (N*M,)
    x_flat = matrix.ravel()
    
    # Create matching y values: each y[i] repeated M times
    y_flat = np.repeat(vector, M)
    
    # Evaluate all points at once
    z_flat = spline.ev(x_flat, y_flat)
    
    return z_flat.reshape(N, M)




#read and interpolate the power spectrum
matter_power_camb = ascii.read(resources+"/test_matterpower.dat")
k, Pk  = matter_power_camb['col1'], matter_power_camb['col2']
# mink =min(k)
# maxk = max(k)
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

dnorm = D1(1.)[0]
def D(a):
    return D1(a)/dnorm

k_grid = np.logspace(-4, 4, 1000)  # Adjust as needed
k_squared = (k_grid**2) / 19.7392
power_spec = PowerSpec(k_grid)
k_grid = k_grid[:, None]

def sigma_squared(z, R):
    window = Window(k_grid * R)
    integrand = k_squared[:, None] * power_spec[:, None] * window**2
    
    # Numerical integration
    sigma2 = np.trapezoid(integrand, k_grid[:, 0], axis=0)
    
    # Apply D(z)^2
    D2 = D(cosmo.scale_factor(z))**2
    return D2 * sigma2

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

def Delta_Sigma_l(z, R, N=100):
    """
    Vectorized Delta_Sigma_l for 1D arrays z and R
    Returns array of shape (len(z), len(R))
    """
    # Create integration grid: (len(R), N)
    x_grid = np.outer(R, np.linspace(0, 1, N))
    x_grid = np.maximum(x_grid, 1e-10)
    
    # Compute Sigma_l for all combinations: (len(z), len(R), N)
    # z[:, None, None] -> (len(z), 1, 1)
    # x_grid[None, :, :] -> (1, len(R), N)
    Sigma_grid = Sigma_l(z[:, None, None], x_grid[None, :, :])
    
    # Integrate over x: (len(z), len(R))
    integrand = x_grid[None, :, :] * Sigma_grid
    integrals = np.trapezoid(integrand, x_grid[None, :, :], axis=-1)
    
    # Final result: (len(z), len(R))
    return 2/(R**2) * integrals - Sigma_l(z[:, None], R[None, :])

def NFW_Delta_Sigma(M200, C200, Z, PCC, SIGMA, M0, radii):
    """
    NFW MODEL:
    
    - M200: mass in Msun
    - C200: concentration
    - Z: redshift
    - PCC: fraction of miscentered signal
    - SIGMA: characteristic miscentering
    - M0: baryonic mass of central gaelaxy
    - radii: a collection of radii at which we calculate the model (mpc)
    
    RETURNS:
    
    A dict containing each part of the signal in Msun/pc2. See below.
    """

    rs = r_vir_m(Z,M200)/C200
    fact =  2*rs*NFW_delta_c(C200)*cosmo.rhoM(Z)
    
    xi = SIGMA/rs
    X = np.outer(radii,1/rs)
  
    sigx= SigX(X)
    sigbarx = SigBar(X)

    signal_BCG = np.outer(M0,1/(np.pi*radii**2))
    signal_BCG = signal_BCG/1e12

    signal_NFW_centre = fact*(sigbarx-sigx)/1e12

    signal_NFW_miscc  = fact*Delta_Sigma_NFW_off_x(X.T,xi).T/1e12
    signal_2ht = (Bias(Z, M200)[:,np.newaxis] * Delta_Sigma_l(Z,radii)).T/1e12
    
    signal_total =signal_BCG + PCC[:,np.newaxis]*signal_NFW_centre.T + (1 - PCC[:,np.newaxis])*signal_NFW_miscc.T + signal_2ht.T
    
    signal = {'Signal': signal_total,
              'BCG Signal': signal_BCG,
              'NFW Signal': signal_NFW_centre.T,
              'Miscentered Signal': signal_NFW_miscc.T,
              'Two-halo term': signal_2ht.T,
              'radii': radii
             }

    return signal
