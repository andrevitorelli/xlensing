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


NFW_delta_c = lambda conc: (cosmo.Cluster_overdensity/3.)*((conc**3)/(np.log(1.+conc)-conc/(1.+conc)))

def r_vir(z,M):
    dc = cosmo.Cluster_overdensity
    result = ((3.*M)/(cosmo.rhocz(z)*4.*np.pi*dc))**(1./3.)
    return result

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
    niu = cosmo.Collapse_density/np.sqrt(sigma_squared(z,radius))
    return niu

#Peak Bias
def Tinker_bias(nu):
    y = np.log10(cosmo.Cluster_overdensity)
    """Tinker et al 2010:"""
    A = 1.0 + 0.24*y*np.exp(-(4/y)**4)
    a = 0.44*y - 0.88
    B = 0.183
    b = 1.5
    C = 0.019 + 0.107*y + 0.19*np.exp(-(4./y)**4.)
    c = 2.4
    result = 1 - A*(nu**a)/(nu**a+cosmo.Collapse_density**a) +B*nu**b +C*nu**c

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
Delta_Sigma_l = lambda z,R: 2/(R*R)*integrate.romberg(lambda x: x*Sigma_l(z,x),
                                                    0,
                                                    R,
                                                    vec_func=True,
                                                    divmax=1)-Sigma_l(z,R)
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
    fact =  2*rs*NFW_delta_c(C200)*cosmo.rhocz(Z)
    
    #adimensional radii
    xi = SIGMA/rs
    X = radii/rs

    sigx= np.array([SigX(r,rs) for r in radii])

    #shear due to the baryonic mass of the central galaxy
    signal_BCG = M0/(np.pi*radii**2)
    signal_BCG = signal_BCG/1e12
    
    #shear due to correctly centered galaxy systems
    signal_NFW_centre =  np.array([(2*fact/(r*r) )*integrate.quad(lambda x: x*SigX(x, rs),0,r,limit=1)[0]  for r in radii])- fact*sigx
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
              'NFW Signal': PCC*signal_NFW_centre,
              'Miscentered Signal': (1-PCC)*signal_NFW_miscc,
              'Two-halo term': signal_2ht,
              'radii': radii
             }

    return signal

def Einasto_shear(Mvir,conc,z,pcc,sigma_off,M0,radii=np.logspace(-1,1,10)):
    return "Not implemented"


