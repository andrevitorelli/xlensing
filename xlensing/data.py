import numpy as np
from . import cosmo

sigmacrit = lambda z_d, z_s: ((cosmo.lightspeed**2.)/(4*np.pi*cosmo.gravity))*cosmo.DA(0,z_s)/(cosmo.DA(0,z_d)*cosmo.DA(z_d,z_s))

#Angular Separation between two points in the sky
def equatorial_to_polar(lon1, lat1, lon2, lat2):
    """ 
    Args
    ----------
    lon1, lat1, lon2, lat2 : Angle, Quantity or float
        Longitude and latitude of the two points.  Quantities should be in
        angular units; floats in radians

    Returns
    -------
    Sep :  float
      Separation in radians
    theta: float
      Azimuth respective to meridian      

    Notes
    -----
    
    The separation part of this fuction was copied directly from astropy to use as a pure function.
    The azimuth part is just a simple implementation of the azimuth formula.
    
    The angular separation is calculated using the Vincenty formula [1]_,
    which is slightly more complex and computationally expensive than
    some alternatives, but is stable at at all distances, including the
    poles and antipodes.

    .. [1] http://en.wikipedia.org/wiki/Great-circle_distance
    """
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
    
def cap_area(radius):
    """Calculates the solid angle of a cap"""
    Omega = 2.0*np.pi*(1-np.cos(radius))*3282.810874*3600 #arcmin^2
    return Omega
def annular_area(rad1,rad2):
    """Calculates the solid angle of a annulus"""
    return cap_area(rad2)-cap_area(rad1)

def lensfit_cluster_lensing(cluster,sources,radius,sys_angle=np.pi/2):
    """
    Gets the lensing signal given a cluster and a lensfit source galaxy catalogue within a 
    radius from which we will get galaxies to measure tangential and cross 
    ellipticities, and the critical lensing density.
    
    Args:
    ----
    
      cluster : a 3-tuple with cluster RA, DEC (in radians) and redshift.
        
      sources : a 7-tuple with:

      - RA, DEC of galaxy in radians
      - galaxy redshift
      - E1, E2: galaxy ellipticity measurements in the CFHT coordinate system.
      - W: Ellipticity measurement weights, given by 1/(intrinsic_ellipticity^2 + ellipticity_uncertainty^2)
      - M: Estimative for the multiplicative bias of the ellipticity measurement.
      
    Returns
    -------
    
      result: a dict of arrays with:
      
      - Critical density: the \Sigma_{crit}, for weak lensing (N values)
      - Tangential Shear: e_t (N values)
      - Cross Shear: e_x (N values)
      - Radial Distance: the angular diameter distance between the center and the position of the obj (N values)
      - Polar Angle: the azimuthal angle for objects in the polar coord. sys. (N values)
      - Weights: the Weights just as from the input #TODO calculate weights here if needed
      - Mult Bias: the multiplicative bias from above
      
    """    
    
    #convert input to np arrays
    cluster = np.array(list(cluster))
    source = np.array(list(sources)).T
    
    #angular diameters
    cluster_DA = cosmo.DA(0,cluster[2])
    angular_radius = radius/cluster_DA
    
    #select cluster area
    region_mask = angular_separation(cluster[0],cluster[1],source[:,0],source[:,1])<  angular_radius
    region = source[region_mask]
    
    #select galaxy backgrounds
    background_condition = (region[:,2]> 1.1*cluster[2] +.1)   #this is contentious and should be changed
    background_region = region[background_condition,:]

    #critical lensing density and polar position of sources/clusters
    sigs = sigmacrit(cluster[2],background_region[:,2])/1e12 #msun/pc^2 is better than msun/mpc^2 for numerical reasons
    rads, theta = equatorial_to_polar(background_region[:,0],
                    background_region[:,1],
                    cluster[0],
                    cluster[1])
    #theta += sys_angle
    #polar ellipticities
    et = -background_region[:,3]*np.cos(2*theta) - background_region[:,4]*np.sin(2*theta)
    ex = -background_region[:,3]*np.sin(2*theta) + background_region[:,4]*np.cos(2*theta)
    
    
    result = {'Critical Density': sigs,
              'Tangential Shear': et,
              'Cross Shear': ex,
              'Radial Distance' : rads*cluster_DA,
              'Polar Angle': theta,
              'Weights': background_region[:,5],
              'Mult. Bias': background_region[:,6],
              'Count': len(background_region)}
    
    return result

def tangential_response(R11, R12, R21, R22,phi):
  Rt = R11*np.cos(2*phi)**2 + R22*np.sin(2*phi)**2 + (R12+R21)*np.sin(2*phi)*np.cos(2*phi)
  return Rt

def metacal_cluster_lensing(cluster,sources,radius,sys_angle=np.pi/2):
    """
    Gets the lensing signal given a cluster and a source galaxy catalogue within a 
    radius from which we will get galaxies to measure tangential and cross 
    ellipticities, and the critical lensing density.
    
    Args:
    ----
    
    cluster = a 3-tuple with cluster RA, DEC (in radians) and redshift.
        
    sources = a 10-tuple with arrays of:

    - 0: RA and
    - 1: DEC of galaxy in radians
    - 2 galaxy redshift
    - 3: E1 and
    - 4: E2 (standard coordinate system RA = -e1)
    - 5: W single opbj measurement weight
    - 6: R11,
    - 7: R12,
    - 8: R21, and
    - 9: R22, the response matrix
      
    Returns
    -------
    
      result: a dict of arrays with:
      
      - Critical density: the \Sigma_{crit}, for weak lensing (N values)
      - Tangential Shear: e_t (N values)
      - Cross Shear: e_x (N values)
      - Radial Distance: the angular diameter distance between the center and the position of the obj (N values)
      - Polar Angle: the azimuthal angle for objects in the polar coord. sys. (N values)
      - Weights: the Weights just as from the input #TODO calculate weights here if needed (N values)
      - Mult Bias: the multiplicative bias from Rt - 1  (N values)
      - count: N   
    
    """   
    
    #convert input to np arrays
    cluster = np.array(list(cluster))
    source = np.array(list(sources)).T
    
    #angular diameters
    cluster_DA = cosmo.DA(0,cluster[2])
    angular_radius = radius/cluster_DA
    
    #select cluster area
    region_mask = equatorial_to_polar(cluster[0],cluster[1],source[:,0],source[:,1])[0]<  angular_radius
    region = source[region_mask]
    
    #select galaxy backgrounds
    background_condition = (region[:,2]> 1.1*cluster[2] +0.2)   #this is contentious and should be changed
    background_region = region[background_condition,:]

    #critical lensing density and polar position of sources/clusters
    sigs = sigmacrit(cluster[2],background_region[:,2])/1e12 #msun/pc^2 is better than msun/mpc^2 for numerical reasons
    rads, theta = equatorial_to_polar(background_region[:,0],
                    background_region[:,1],
                    cluster[0],
                    cluster[1])
    theta += sys_angle
    #polar ellipticities
    et = -background_region[:,3]*np.cos(2*theta) - background_region[:,4]*np.sin(2*theta)
    ex = -background_region[:,3]*np.sin(2*theta) + background_region[:,4]*np.cos(2*theta)
    
    Rt = tangential_response(background_region[:,6],
                             background_region[:,7],
                             background_region[:,8],
                             background_region[:,9], theta)
    
    result = {'Critical Density': sigs,
              'Tangential Shear': et,
              'Cross Shear': ex,
              'Radial Distance' : rads*cluster_DA,
              'Polar Angle': theta,
              'Weights': background_region[:,5],
              'Mult. Bias': Rt - 1,
              'Count': len(background_region)}
    
    return result#, background_region

def signal(stake,bin_limits):
  """
  cluster_backgrounds = a list of ndarrays, each containing cluster background galaxies for lensing. 
  
  They should contain: 
  - (0) Sigma_crit: the critical density calculated from the cluster and galaxy redshifts
  - (1) e_t: the tangential component of the shear
  - (2) e_x: the cross component of the shear
  - (3) W: the weight of the ellipticity measurement
  - (4) R: the angular diameter radius in Mpc/h between the cluster centre and the background
        galaxy position.
  - (5) M: the estimation of multiplicative biases 
  """


  Nbins=len(bin_limits)
  Delta_Sigmas = np.empty(Nbins) #E-mode signal (tang. shear)
  Delta_Xigmas = np.empty(Nbins) #B-mode signal (cross shear)

  for radius in range(Nbins):
    #populate radial bins
    bin_upper_cut = stake[:,stake[4,:]<bin_limits[radius,1]]
    bin_cut = bin_upper_cut[:,bin_upper_cut[4,:]>bin_limits[radius,0]]

    #sigma = average sigma_crit * shear * weight=(W/sigma_crit^2)
    Sigma = np.average(bin_cut[0,:]*bin_cut[1,:],weights= bin_cut[3,:]/(bin_cut[0,:]**2))
    Xigma = np.average(bin_cut[0,:]*bin_cut[2,:],weights= bin_cut[3,:]/(bin_cut[0,:]**2))

    #average multiplicative bias correction
    One_plus_K = np.average(bin_cut[5,:]+1,weights= bin_cut[3,:]/(bin_cut[0,:]**2))

    Delta_Sigmas[radius] = Sigma/One_plus_K
    Delta_Xigmas[radius] = Xigma/One_plus_K

  return Delta_Sigmas, Delta_Xigmas


def stacked_signal(cluster_backgrounds,bin_limits,Nboot=200):
    """
    cluster_backgrounds = a list of ndarrays, each containing 
    cluster background galaxies for lensing. They should contain: 
    - (0) Sigma_crit: the critical density calculated from the cluster and galaxy redshifts
    - (1) e_t: the tangential component of the shear
    - (2) e_x: the cross component of the shear
    - (3) W: the weight of the ellipticity measurement
    - (4) R: the angular diameter radius in Mpc/h between the cluster centre and the background
          galaxy position.
    - (5) M: the estimation of multiplicative biases
    
    bin_limits = an array containing the bin lower and upper bounds
    
    Nboot = the number of resamplings desired
    
    """
    
    print("Total galaxies available per bin:")
    sources_radii = np.hstack([cluster_backgrounds[i][4] for i in range(len(cluster_backgrounds))])
    bin_counts = np.array([len(sources_radii[(sources_radii > bini[0]) & (sources_radii < bini[1])]) for bini in bin_limits]) 
    total_gals = sum(bin_counts)
    print(bin_counts)
    print()
    #calculate boost factors


    max_radius = np.max(bin_limits)
    min_radius = np.min(bin_limits)
    
    area =np.pi*(max_radius**2-min_radius**2)#flat sky
    density = total_gals/area
    RR_area = 4*max_radius**2
    RR_gals = density*RR_area
    RRx=np.random.uniform(-max_radius,max_radius,round(RR_gals))
    RRy=np.random.uniform(-max_radius,max_radius,round(RR_gals))
    RR=np.sqrt(RRx**2+RRy**2)
    RR_bins = np.array([len(RR[(RR > bini[0]) & (RR < bini[1])]) for bini in bin_limits ] )
    boosts=bin_counts/RR_bins
    print("Boost factors")
    print(boosts)

    
    
    Nbins=len(bin_limits)
    #sorts Nboot selections of the clusters all at once
    resample=np.random.randint(0,len(cluster_backgrounds),(Nboot,len(cluster_backgrounds)))

    Delta_Sigmas = np.empty((Nboot,Nbins)) #E-mode signal (tang. shear)
    Delta_Xigmas = np.empty((Nboot,Nbins)) #B-mode signal (cross shear)

    #bootstrap
    import tqdm  
    for sampleNo in tqdm.tqdm(range(len(resample))):
      stake = np.hstack([cluster_backgrounds[i] for i in resample[sampleNo]])

      sigmas, xigmas = signal(stake,bin_limits)
      Delta_Sigmas[sampleNo] = sigmas
      Delta_Xigmas[sampleNo] = xigmas

    #gather results
    sigmas = np.mean(Delta_Sigmas,axis=0)
    xigmas = np.mean(Delta_Xigmas,axis=0)

    sigmas_cov = np.cov(Delta_Sigmas.T)
    xigmas_cov = np.cov(Delta_Xigmas.T)

    return sigmas*boosts, sigmas_cov, xigmas, xigmas_cov

def single_cluster(cluster_backgrounds,bin_limits,Nboot=500):
    """
    cluster_backgrounds = a list of ndarrays, each containing 
    cluster background galaxies for lensing. They should contain: 
    - (0) Sigma_crit: the critical density calculated from the cluster and galaxy redshifts
    - (1) e_t: the tangential component of the shear
    - (2) e_x: the cross component of the shear
    - (3) W: the weight of the ellipticity measurement
    - (4) R: the angular diameter radius in Mpc/h between the cluster centre and the background
          galaxy position.
    - (5) M: the estimation of multiplicative biases
    
    bin_limits = an array containing the bin lower and upper bounds
    
    Nboot = the number of resamplings desired
    
    TODO: invert resampling bin structure in 1-cluster stacks
    """
    
    print("Total galaxies available per bin:")
    sources_radii = np.hstack([cluster_backgrounds[i][4] for i in range(len(cluster_backgrounds))])
    print([len(sources_radii[(sources_radii > bini[0]) & (sources_radii < bini[1])]) for bini in bin_limits ] )
    print()    
    
    Nbins=len(bin_limits)
    
    ##for single cluster stacks

    print("Single cluster:")
    background = cluster_backgrounds[0]
    print("Separating galaxies per radial bin...")
    #separate all available galaxies into radial bins
    radial_background = []

    for bins in bin_limits:
        bin_upper_cut  = background[:,background[4]<bins[1]]
        bin_lower_cut  = bin_upper_cut[:,bin_upper_cut[4]>bins[0]]
        radial_background.append(bin_lower_cut)

    Delta_Sigmas = np.empty((Nboot,Nbins)) #sigma_crit * et (E-mode signal)
    Delta_Xigmas = np.empty((Nboot,Nbins)) #sigma_crit * ex (B-mode signal)

    #bootstrap
    for sampleNo in range(Nboot):

        for radius, radial_bin in enumerate(radial_background):

            sorted_galaxies = np.random.randint(0,len(radial_bin.T),len(radial_bin.T))
            sorted_bin = np.array([radial_bin.T[i] for i in sorted_galaxies]).T

            Sigma = np.average(sorted_bin[0,:]*sorted_bin[1,:],weights= sorted_bin[3,:]/(sorted_bin[0,:]**2))
            Xigma = np.average(sorted_bin[0,:]*sorted_bin[2,:],weights= sorted_bin[3,:]/(sorted_bin[0,:]**2))

            #average multiplicative bias correction
            One_plus_K = np.average(sorted_bin[5,:]+1,weights= sorted_bin[3,:]/(sorted_bin[0,:]**2))

            Delta_Sigmas[sampleNo,radius] = Sigma/One_plus_K
            Delta_Xigmas[sampleNo,radius] = Xigma/One_plus_K

    Delta_Sigmas = np.array(Delta_Sigmas)
    Delta_Xigmas = np.array(Delta_Xigmas)

    #gather results
    sigmas = np.mean(Delta_Sigmas,axis=0)
    xigmas = np.mean(Delta_Xigmas,axis=0)

    sigmas_cov = np.cov(Delta_Sigmas.T)
    xigmas_cov = np.cov(Delta_Xigmas.T)
    
    return sigmas, sigmas_cov, xigmas, xigmas_cov


