from astropy.cosmology import FlatLambdaCDM
from numpy import sqrt


#Constants
lightspeed = 299792.458  #km/s
cluster_overdensity = 200
collapse_density = 1.686
gravity = 4.302e-9
h=1

#Default cosmological parameters
OmegaM = 0.3
OmegaK = 0.0
cosmology = FlatLambdaCDM(H0=100.*h,Om0=OmegaM) #can be changed in the code if wanted
DA = lambda x,y: cosmology.angular_diameter_distance_z1z2(x,y).value #Mpc
H = lambda a: 100.*h*sqrt(OmegaM*(a)**(-3) + OmegaK*(a)**(-2) + 1.-OmegaM)

#Scale factor from redshifts
scale_factor = lambda z: 1./(1.+z)

#Critical density of the Universe at redshift Z
rhocz = lambda z: 27746677.666*(H(scale_factor(z))**2)#3/8piG in Msun, kms, mpc

#Critical mass density of the universe at redshift z
rhoM = lambda z: OmegaM*rhocz(z)
