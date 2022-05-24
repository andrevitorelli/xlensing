import numpy as np
from . import cosmo


sigmacrit = lambda z_d, z_s: ((cosmo.lightspeed**2.)/(4*np.pi*cosmo.gravity))*cosmo.DA(0,z_s)/(cosmo.DA(0,z_d)*cosmo.DA(z_d,z_s))


#Angular Separation between two points in the sky
def angular_separation(lon1, lat1, lon2, lat2):
    """
    Parameters
    ----------
    lon1, lat1, lon2, lat2 : Angle, Quantity or float
        Longitude and latitude of the two points.  Quantities should be in
        angular units; floats in radians

    Returns
    -------
    angular separation : Quantity or float
        Type depends on input; Quantity in angular units, or float in radians

    Notes
    -----
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
    return sep

def equatorial_to_polar(RA,Dec,RA_center,Dec_center,sys_angle=np.pi/2): 
    """
    Converts from an equatorial system of coordinates to a
    polar system containing an azimuthal angle and a separation from
    a reference point.    
    """
    
    
    #center all points in RA
    RAprime = RA-RA_center
    RA_center=0
    #convert to positive values of RA
    negative = (RAprime<0)
    RAprime[negative] = 2*np.pi+RAprime[negative]
    #define wich quadrant is each point
    Qd1 = (RAprime<np.pi)&(Dec-Dec_center>0)
    Qd2 = (RAprime<np.pi)&(Dec-Dec_center<0)
    Qd3 = (RAprime>np.pi)&(Dec-Dec_center<0)
    Qd4 = (RAprime>np.pi)&(Dec-Dec_center>0)
    
    #Calculate the distance between the center and object, and the azimuthal angle
    Sep = angular_separation(RAprime, Dec, RA_center,Dec_center)
        
    #build a triangle to calculate the spherical cosine law
    x = angular_separation(RA_center,Dec,RA_center,Dec_center)
    y = angular_separation(RAprime,Dec, RA_center,Dec)
    
    #Apply shperical cosine law
    cosT = (np.cos(y) - np.cos(x)*np.cos(Sep))/(np.sin(x)*np.sin(Sep))
    #Round cosines that went over because of rounding errors
    roundinghigh = (cosT >  1)
    roundinglow  = (cosT < -1)
    cosT[roundinghigh]=1
    cosT[roundinglow]=-1
    uTheta = np.arccos(cosT)
    #Correct the angle for quadrant (because the above law calculates the acute angle between the "horizontal-RA" direction
    #and the angular separation great circle between the center and the object
    Theta= np.zeros(len(uTheta))
    
    Theta[Qd1] = uTheta[Qd1]
    Theta[Qd2] = np.pi-uTheta[Qd2] 
    Theta[Qd3] = np.pi+uTheta[Qd3]
    Theta[Qd4] = 2*np.pi-uTheta[Qd4]
    
    Theta = Theta+sys_angle
    return Sep, Theta
    
def cap_area(radius):
    """Calculates the solid angle of a cap"""
    Omega = 2.0*np.pi*(1-np.cos(radius))*3282.810874*3600 #arcmin^2
    return Omega
def annular_area(rad1,rad2):
    """Calculates the solid angle of a annulus"""
    return cap_area(rad2)-cap_area(rad1)

def cluster_lensing(cluster,sources,radius,sys_angle=np.pi/2):
    """
    Gets the lensing signal given a cluster and a source galaxy catalogue within a 
    radius from which we will get galaxies to measure tangential and cross 
    ellipticities, and the critical lensing density.
    
    cluster = a 3-tuple with cluster RA, DEC (in radians) and redshift.
        
    sources = a 7-tuple with:
    
    - RA, DEC of galaxy in radians
    - galaxy redshift
    - E1, E2: galaxy ellipticity measurements in the CFHT coordinate system.
    - W: Ellipticity measurement weights, given by 1/(intrinsic_ellipticity^2 + ellipticity_uncertainty^2)
    - M: Estimative for the multiplicative bias of the ellipticity measurement.
    """
    
    
    #convert input to np arrays
    cluster_RA, cluster_DEC, cluster_z = cluster
    cluster = np.array([cluster_RA,cluster_DEC,cluster_z]).T

    source_RA, source_DEC, source_z, source_E1, source_E2, source_W, source_M = sources
    source = np.array([source_RA,source_DEC,source_z,source_E1,source_E2,source_W,source_M]).T
    
    #angular diameters
    cluster_DA = cosmo.DA(0,cluster_z)
    angular_radius = radius/cluster_DA
    
    #select cluster area
    region_mask = angular_separation(cluster_RA,cluster_DEC,source[:,0],source[:,1])<  angular_radius
    region = source[region_mask]
    
    #select galaxy backgrounds
    background_condition = (region[:,2]> 1.1*cluster_z +0.2)   #this is contentious and should be changed
    background_region = region[background_condition,:]

    #critical lensing density and polar position of sources/clusters
    sigs = sigmacrit(cluster_z,background_region[:,2])/1e12 #msun/pc^2 is better than msun/mpc^2 for numerical reasons
    rads, theta = equatorial_to_polar(background_region[:,0],
                    background_region[:,1],
                    cluster_RA,
                    cluster_DEC,sys_angle)

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

def stack(cluster_backgrounds,bin_limits,Nboot=200):
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
    if len(cluster_backgrounds) == 1:
        background = cluster_backgrounds[0]
        countgals = len(background.T)

        #separate all available galaxies into radial bins
        radial_background = []
        print(background)
        for bins in bin_limits:
            bin_upper_cut  = background[:,background[4]<bins[1]]
            bin_lower_cut  = bin_upper_cut[:,bin_upper_cut[4]>bins[0]]
            radial_background.append(bin_lower_cut)
            
        Delta_Sigmas = np.empty((Nboot,Nbins)) #sigma_crit * et (E-mode signal)
        Delta_Xigmas = np.empty((Nboot,Nbins)) #sigma_crit * ex (B-mode signal)
        
        for sampleNo in range(Nboot):

            for radius, radial_bin in enumerate(radial_background):
                sorted_galaxies = np.random.randint(0,len(radial_bin),len(radial_bin))
                sorted_bin = np.array([radial_bin[i] for i in sorted_galaxies])
               
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

    else:
        #sorts Nboot selections of the clusters all at once
        resample=np.random.randint(0,len(cluster_backgrounds),(Nboot,len(cluster_backgrounds)))
        
        Delta_Sigmas = np.empty((Nboot,Nbins)) #E-mode signal (tang. shear)
        Delta_Xigmas = np.empty((Nboot,Nbins)) #B-mode signal (cross shear)
        
        #bootstrap
        resamples = arange(Nboot)
        
        def resampler(resample):
            stake = np.hstack([cluster_backgrounds[i] for i in resample[sampleNo]])

            for radius in range(Nbins):
                #populate radial bins
                bin_upper_cut = stake[:,stake[4,:]<bin_limits[radius,1]]
                bin_cut = bin_upper_cut[:,bin_upper_cut[4,:]>bin_limits[radius,0]]
                
                #sigma = average sigma_crit * shear * weight=(W/sigma_crit^2)
                Sigma = np.average(bin_cut[0,:]*bin_cut[1,:],weights= bin_cut[3,:]/(bin_cut[0,:]**2))
                Xigma = np.average(bin_cut[0,:]*bin_cut[2,:],weights= bin_cut[3,:]/(bin_cut[0,:]**2))
                
                #average multiplicative bias correction
                One_plus_K = np.average(bin_cut[5,:]+1,weights= bin_cut[3,:]/(bin_cut[0,:]**2))

                Delta_Sigmas[resample,radius] = Sigma/One_plus_K
                Delta_Xigmas[resample,radius] = Xigma/One_plus_K

            del stake
        
        pool = Pool(cpu_count()) 
        pool.map(resampler, resamples)
        pool.close()
            
        #gather results
        sigmas = np.mean(Delta_Sigmas,axis=0)
        xigmas = np.mean(Delta_Xigmas,axis=0)
        
        sigmas_cov = np.cov(Delta_Sigmas.T)
        xigmas_cov = np.cov(Delta_Xigmas.T)
    

    return sigmas, sigmas_cov, xigmas, xigmas_cov



  