"""Module for simulating/injecting simulated clusters on galaxy catalogs."""

import numpy as np
import ngmix
from xlensing.cosmo import DA, cosmology,cluster_overdensity,lightspeed,gravity,OmegaM
from xlensing.model import 

def rhoM(z):
  """Mass fraction of the critical mass density of the universe"""
  mass_density = OmegaM*cosmology.critical_density(z).to('Msun/Mpc**3').value
  return mass_density

def r_vir(zlens, Mlens):
  """Virial radius of an spherical halo of mass Mlens at redshift zlens"""
  radius = ((3.*Mlens)/(rhoM(zlens)*4.*np.pi*cluster_overdensity))**(1./3.)
  return radius

def NFW_delta_c(conc):
  """relationship between the concentration and the NFW mass distribution"""
  deltac = (cluster_overdensity/3.)*((conc**3)/(np.log(1.+conc)-conc/(1.+conc)))
  return deltac

def critical_density(zlens, zsource):
  """Critical density"""
  sigmacrit = (lightspeed**2.)/(4*np.pi*gravity)*DA(0,zsource)/(DA(0,zlens)*DA(zlens,zsource))
  return np.heaviside(zsource,zlens)*sigmacrit

critical_density = np.vectorize(critical_density)

def gNFW(x):
  """shear caused by an NFW profile in adimensional radii"""
  if x<1:
    g = 8*np.arctanh(np.sqrt((1-x)/(1+x)))/(x**2*np.sqrt(1-x**2)) + 4/(x**2)*np.log(x/2) - 2/(x**2-1) + 4*np.arctanh(np.sqrt((1-x)/(1+x)))/( (x**2-1)*np.sqrt(1-x**2) )
  if x>1:
    g = 8*np.arctan (np.sqrt((x-1)/(1+x)))/(x**2*np.sqrt(x**2-1)) + 4/(x**2)*np.log(x/2) - 2/(x**2-1) + 4*np.arctan (np.sqrt((x-1)/(1+x)))/( (x**2-1)*np.sqrt(x**2-1) )
  if x ==1:
    g = 10/3 +4*np.log(1/2)
  return g

def NFW_tangential_shear(Mlens, conc, zlens, zsource, r):
  """NFW shear in physical coordinates at the plane of the lens"""  
  #define scale radius of NFW
  rs= r_vir(zlens,Mlens)/conc
  #Rescaled radiii
  x = r/rs
  #critical surface density
  sigcrit = critical_density(zlens,zsource)  
  #total shear caused by NFW profile (Wright, Brainerd, 99)
  gammat = (rs*NFW_delta_c(conc)*rhoM(zlens)/sigcrit)*gNFW(x)
  return gammat

NFW_tangential_shear = np.vectorize(NFW_tangential_shear)

def equatorial_to_polar(lon1, lat1, lon2, lat2):
  #vincenty's formula
  sdlon = np.sin(lon2 - lon1)
  cdlon = np.cos(lon2 - lon1)
  slat1 = np.sin(lat1)
  slat2 = np.sin(lat2)
  clat1 = np.cos(lat1)
  clat2 = np.cos(lat2)
  num1 = clat2 * sdlon
  num2 = clat1 * slat2 - slat1 * clat2 * cdlon
  denominator = slat1 * slat2 + clat1 * clat2 * cdlon
  sep = np.arctan2(np.sqrt(num1 ** 2 + num2 ** 2), denominator)

  ###Azimuth formula
  L = lon2-lon1
  denominator2 = clat1*slat2/clat2 - slat1*cdlon
  theta = np.arctan2(sdlon,denominator2)
  return sep, theta

def NFW_wcs_cluster_shear(Mlens,conc,zlens,zsource,cluster_RA,cluster_DEC,source_RA,source_DEC):
  """NFW shear in the sky"""
  #compute angular diameter distance
  cluster_DA = DA(0,zlens)
  #compute separation and azimuth
  sep, theta = equatorial_to_polar(cluster_RA,cluster_DEC,source_RA,source_DEC)
  #get physical separation
  Radius = sep*cluster_DA
  #get tangential shear
  gammat = NFW_tangential_shear(Mlens, conc, zlens, zsource, Radius)
  #convert into complex shear
  g = gammat*np.exp(2.j*(theta)) #this theta gives the RA -DEC orientation for e1, e2
  return g

def add_shears(e,eadd):
  """sum shears"""
  sume = (e+eadd)/(1+np.conj(eadd)*e)  
  return sume

def make_simple_random_cat(density, width_rad, zrange, shape_noise,seed=1):
  """simple random galaxies with a metacal-catalog like output"""
  rng = np.random.RandomState(seed)

  width_arcmin = width_rad*3437.75
  Ngals = round(density*width_arcmin**2)
  
  sources_RA = rng.uniform(-width_rad,width_rad,size=Ngals)
  sources_DEC = rng.uniform(-width_rad,width_rad,size=Ngals)
  sources_Z = rng.uniform(zrange[0],zrange[1],size=Ngals)#np.random.uniform(0.3,2.,1000)
  ellip = 2.
  
  sources_E1, sources_E2 = ngmix.priors.shape.GPriorBA(sigma=shape_noise,rng=rng).sample2d(Ngals)
  sources_R11 = rng.normal(1,shape_noise/100,size=Ngals)
  sources_R12 = rng.normal(0,shape_noise/100,size=Ngals)
  sources_R21 = rng.normal(0,shape_noise/100,size=Ngals)
  sources_R22 = rng.normal(1,shape_noise/100,size=Ngals)
  sources_W = np.ones(Ngals)
  simplecat = np.array([
    sources_RA,
    sources_DEC,
    sources_Z, 
    sources_E1, 
    sources_E2, 
    sources_R11, 
    sources_R12, 
    sources_R21, 
    sources_R22, 
    sources_W])
  return simplecat

def apply_NFW_shear_region(cluster,galaxies):
  """apply NFW shear over a catalog of galaxies
  
  cluster: RA, DEC, z, M200, C200
  
  galaxies: ndarray with:
  
  - RA
  - DEC
  - z
  - e1
  - e2   
  """
  gammas = []

  for galaxy in galaxies:
    if galaxy[2]> cluster[2]: #redshift comparison
      gammas += [NFW_wcs_cluster_shear(
        cluster[3],#cluster mass
        cluster[4],#concentration
        cluster[2], #z cluster
        galaxy[2], #z galaxy
        cluster[0],cluster[1], #cluster center
        galaxy[0],galaxy[1] #galaxy position
    )]
    else:
      gammas += [0]
  
  gammas=np.array(gammas)

  sheared_e = add_shears(galaxies[:,3]+1.j*galaxies[:,4],gammas)
  
  galaxies[:,3] = sheared_e.real
  galaxies[:,4] = sheared_e.imag
  
  return galaxies