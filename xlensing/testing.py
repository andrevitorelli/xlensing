import numpy as np
from . import cosmo, model, data

def truncnorm(low, high, loc, sigma, size): #I hate scipy truncnorm
    x = []
    while len(x)<size:
        xi = np.random.normal(loc,sigma)
        if xi < high and xi > low:
            x.append(xi)
    return np.array(x)

def gen_gal(Ngals=1e5,Zcluster=.3,rmax=12., gmax=.6,gspread=.1):
  Ngals = int(Ngals)
  Zgals = np.random.uniform(0,1,size=int(Ngals))

  angular_distance = cosmo.cosmology.angular_diameter_distance(Zcluster).value #Mpc
  r_galaxies = np.random.uniform(0,rmax,size=Ngals) #Mpc
  theta_galaxies = np.random.uniform(0,2*np.pi, size=Ngals)

  RAgals = r_galaxies*np.cos(theta_galaxies)/angular_distance
  DECgals= r_galaxies*np.sin(theta_galaxies)/angular_distance

  #intrinsic ellipticity of the galaxies
  Ebarr = 0.1
  E1gals = truncnorm(-gmax,gmax,0,Ebarr,Ngals)
  E2gals = truncnorm(-gmax,gmax,0,Ebarr,Ngals)
  return E1gals, E2gals, RAgals, DECgals, Zgals
  
  
def gNFW(x):
  """
  Expected reduced shear of an NFW profile
  """
  if x<1:
      g = 8*np.arctanh(np.sqrt((1-x)/(1+x)))/(x**2*np.sqrt(1-x**2)) + 4/(x**2)*np.log(x/2) - 2/(x**2-1) + 4*np.arctanh(np.sqrt((1-x)/(1+x)))/( (x**2-1)*np.sqrt(1-x**2) )
  if x>1:
      g = 8*np.arctan(np.sqrt((x-1)/(1+x)))/(x**2*np.sqrt(x**2-1)) + 4/(x**2)*np.log(x/2) - 2/(x**2-1) + 4*np.arctan (np.sqrt((x-1)/(1+x)))/( (x**2-1)*np.sqrt(x**2-1) )
  if x ==1:
      g = 10/3 +4*np.log(1/2)
  return g


  
def NFW_shear(cluster, galaxies):
  """
  NFW shear

  Parameters:
  ----------
  cluster: tuple of floats

    Mvir: mass in solar masses
    conc: concentration
    centre_RA: Right Ascention in radians
    centre_DEC: Declination in radians
    z_lens: redshift of the cluster

  galaxies: tuple of numpy arrays
    gal_RA: Right Ascention in radians for each galaxy
    gal_DEC: Declination in radians for each galaxy
    z_galaxy: redshift for each galaxy
    e1, e2: intrinsic shape of each galaxy


  Returns:
  epsilon: complex
    Ellipticity of galaxies after shear    
  """

  #unpack data
  Mvir, conc, centre_RA, centre_DEC, z_lens = cluster
  gal_RA, gal_DEC, z_galaxy, e1, e2  = galaxies

  #find physical radius in the plane of the sky
  sep, theta = data.equatorial_to_polar(gal_RA, gal_DEC, centre_RA, centre_DEC)
  ang_dist = cosmo.cosmology.angular_diameter_distance(z_lens).value
  r = sep*ang_dist
  theta= theta +np.pi/2 #the polar angle is 90º different from the tangent (I took 6 years to figure this out)

  #get complex shape
  intr_shape = e1 + e2*1.j

  Ngals = len(intr_shape)

  #define scale radius of NFW
  rs= model.r_vir(z_lens,Mvir)/conc

  #NFW kappa factor
  fact =  rs*model.NFW_delta_c(conc)*cosmo.rhoM(z_lens)

  #Rescaled radiii
  x = r/rs

  #critical surface density
  sigcrit = data.sigmacrit(z_lens,z_galaxy)


  #total shear caused by NFW profile (Wright, Brainerd, 99)
  gammat = (fact/sigcrit)*np.array([gNFW(r) for r in x]) #add 1% error to measurements

  #complex gamma, reduced shear
  gamma = gammat*np.exp(2.j*(theta))

  #gt = gamma/(1+kappa)

  #final shape
  epsilon = np.empty(Ngals,dtype='complex')
  for i in range(Ngals):
      if z_galaxy[i] > z_lens:
          epsilon[i] =  intr_shape[i] + gamma[i]#
      else:
          epsilon[i] =  intr_shape[i]
  return epsilon