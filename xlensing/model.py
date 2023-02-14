# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:50:50 2015

@author: andrezvitorelli
"""

########################################Model Basics############################
#Radius as a function of mass for a given mass density at redshift z
import numpy as np
from . import cosmo
from astropy.table import Table
from scipy.interpolate import UnivariateSpline as USpline
from scipy.interpolate import RectBivariateSpline as BSpline
from astropy.io import fits, ascii
from scipy import integrate

from pathlib import Path

resources = str(Path(__file__).parent)


def NFW_delta_c(conc):
  if conc >0: 
    return (cosmo.cluster_overdensity/3.)*((conc**3)/(np.log(1.+conc)-conc/(1.+conc)))

  return 1e-12

def r_vir_c(z,M):
    dc = cosmo.cluster_overdensity #\Delta (standard = 200)
    result = ((3.*M)/(cosmo.rhocz(z)*4.*np.pi*dc))**(1./3.)
    return result

def r_vir_m(z,M):
    dc = cosmo.cluster_overdensity #\Delta (standard = 200)
    result = ((3.*M)/(cosmo.rhoM(z)*4.*np.pi*dc))**(1./3.)
    return result

r_vir = r_vir_m
rhoz = cosmo.rhoM


#call the C library for NFW profile shape
from ctypes import CDLL, c_double
lib = CDLL(resources+"/sigma.so")
SigX = lib.SigX
P_off= lib.P_off
SigXr = lib.SigXr

#set the parameters to be passed to functions
SigX.restype = c_double
SigX.argtypes = [c_double,c_double]
P_off.restype = c_double
P_off.argtypes = [c_double,c_double]
SigXr.restype = c_double
SigXr.argtypes = [c_double,c_double,c_double,c_double]


#load NFW miscentered profile
MscRadii = Table.read(resources+"/misc_NFW_radii.fits")
Profs = Table.read(resources+"/misc_NFW_profiles.fits")

print("Lookup table loaded!")
Delta_Sigma_NFW_off_x = BSpline(MscRadii['col0'],Profs['col0'],Profs['col1'].T,s=0)

#read and interpolate the power spectrum
matter_power_camb = ascii.read(resources+"/test_matterpower.dat")
k, Pk  = matter_power_camb['col1'], matter_power_camb['col2']
mink =min(k)
maxk = max(k)
PowerSpec = USpline(k,Pk,s=0,k=5,ext=1)

#read and interpolate matter correlation
matt_corr_table = Table.read(resources+"/matt_corr.fits")
raios_matt, matt_cor = matt_corr_table['col0'], matt_corr_table['col1']
matter_correlation_norm = USpline(raios_matt,
                                  np.array(matt_cor)/0.6434659,
                                  s=0,k=5, ext=1)#0.802**2

#Window function
Window = lambda x: (3/(x*x*x))*(np.sin(x)-x*np.cos(x))

#Growth function
def D1(a):
    return  (cosmo.H(a))*integrate.quad(lambda x: 1/(x*cosmo.H(x)/(100*cosmo.h))**3,
                                  0,
                                  a,
                                  limit=1,full_output=1)[0]

#Normalised Growth function
def D(a):
    return D1(a)/D1(1.)

#Sigma^2(r,Z)
def sigma_squared(z,R):
    sigma2 = (D(cosmo.scale_factor(z))**2)*integrate.quad(
        lambda k: ((k*k)/(19.7392))*PowerSpec(k)*(Window(k*R))*(Window(k*R)),
        0,
        np.inf,
        limit=1,full_output=1)[0]#2*pi**2= 19.7392...
    return sigma2

#Sigma_8
def sigma8(*args):
    return np.sqrt(sigma_squared(0,8))

#Peak (delta_c/sigma)
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

#Implement Johnston 2007 2ht
w_johnston_val = Table.read(resources+"/W_johnston.fits") #precalculated for fast interpolation function.
W_Johnston = USpline(w_johnston_val['col0'],w_johnston_val['col1'],s=0,k=5,ext=1)
Sigma_l = lambda z,R: (1+z)**2*W_Johnston((1+z)*R)
Delta_Sigma_l = lambda z,R: 2/(R*R)*integrate.quad(lambda x: x*Sigma_l(z,x),
                                                    0,
                                                    R,
                                                    limit=1,full_output=1)[0]-Sigma_l(z,R)
Delta_Sigma_l= np.vectorize(Delta_Sigma_l)

def NFW_shear(M200, C200, Z, PCC, SIGMA, M0, radii):
    """
    NFW MODEL:
    
    - M200: mass in Msun
    - C200: concentration
    - Z: redshift
    - PCC: fraction of miscentered signal
    - SIGMA: characteristic miscentering
    - M0: baryonic mass of central gaelaxy
    - radii: a collection of radii at which we calculate the model
    
    RETURNS:
    
    A dict containing each part of the signal. See below.
    """
    
    #discard absurd values that may be proposed by MCMC-like algorithms
    if PCC > 1 or PCC < 0:
        PCC = float('NaN')
    
    
    #scale radius
    rs= r_vir(Z,M200)/C200
    
    #dimensional factors
    fact =  2*rs*NFW_delta_c(C200)*rhoz(Z)
    
    #adimensional radii
    xi = SIGMA/rs
    X = radii/rs

    sigx= np.array([SigX(r,rs) for r in radii])

    #shear due to the baryonic mass of the central galaxy
    signal_BCG = M0/(np.pi*radii**2)
    signal_BCG = signal_BCG/1.e12
    
    #shear due to correctly centered galaxy systems
    signal_NFW_centre =  np.array([(2*fact/(r*r) )*integrate.quad(lambda x: x*SigX(x, rs),
                                                                  0,
                                                                  r,epsabs=0.1,epsrel=0.1,limit=1,full_output=1)[0]  for r in radii])- fact*sigx
    signal_NFW_centre =  signal_NFW_centre/1e12

    #shear due to miscentered galaxy systems
    try:
        signal_NFW_miscc  = fact*Delta_Sigma_NFW_off_x(X, xi).reshape(radii.shape)/1e12
    except:
        signal_NFW_miscc = float('NaN')
        

    #shear due to the large scale structure of the universe
    signal_2ht = Delta_Sigma_l(Z, radii)*Bias(Z, M200)/1e12

    signal_total = signal_BCG + PCC*signal_NFW_centre + (1 - PCC)*signal_NFW_miscc + signal_2ht
    
    signal = {'Signal': signal_total,
              'BCG Signal': signal_BCG,
              'NFW Signal': signal_NFW_centre,
              'Miscentered Signal': signal_NFW_miscc,
              'Two-halo term': signal_2ht,
              'radii': radii
             }

    return signal
  
def _form(x):
  if x > 1:
    sq = np.sqrt(x**2-1)
    result = np.arctan(sq)/sq
  if x < 1:
    sq = np.sqrt(1-x**2)
    result = np.arctanh(sq)/sq
  if x == 1:
    result = 1  
  return result
    
_form = np.vectorize(_form)  
from scipy.interpolate import interp1d
xx = np.linspace(1e-2,100,1000000)
_form = interp1d(xx,_form(xx))

def Boost_model(B0,RS,radii):
  """From McClintock et al. 2018"""
  x = radii/RS
  boosts = 1 + B0 *(1-_form(x))/(x*x-1)
  return boosts 
  

def Einasto_shear(Mvir,conc,z,pcc,sigma_off,M0,radii=np.logspace(-1,1,10)):
  return "Not implemented"


def c_DuttonMaccio(z, m,c=1., h=1.):
  """Concentration from c(M) relation in Dutton & Maccio (2014).
  
  Code taken from Jes Ford on github.
  
  Parameters
  ----------
  z : float or array_like
      Redshift(s) of halos.
  m : float or array_like
      Mass(es) of halos (m200 definition), in units of solar masses.
  h : float, optional
      Hubble parameter. Default is from Planck13.
  Returns
  ----------
  ndarray
      Concentration values (c200) for halos.
  References
  ----------
  Calculation from Planck-based results of simulations presented in:
  A.A. Dutton & A.V. Maccio, "Cold dark matter haloes in the Planck era:
  evolution of structural parameters for Einasto and NFW profiles,"
  Monthly Notices of the Royal Astronomical Society, Volume 441, Issue 4,
  p.3359-3374, 2014.
  """


  a = 0.52 + 0.385 * np.exp(-0.617 * (z**1.21))  # EQ 10
  b = -0.101 + 0.026 * z                         # EQ 11

  logc200 = a + b * np.log10(m * h / (c*1e12))  # EQ 7 modified

  concentration = 10.**logc200
  return concentration
  
def mass_lambda_McClintock18(Lambda,z):
  #pivots
  log10Lambda0 = np.log10(40)
  log10z0 = np.log10(0.35)
  log10M0 = 14.489
  #scalings
  Flambda = 1.356
  Gz = -0.30
    
  mlog10M200 = log10M0 + Flambda*(np.log10(Lambda)-log10Lambda0) + Gz*(np.log10(z) - log10z0)
    
  return mlog10M200