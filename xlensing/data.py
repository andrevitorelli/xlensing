import numpy as np
from . import cosmo


sigmacrit = lambda z_d, z_s: ((cosmo.lightspeed**2.)/(4*np.pi*cosmo.Gravity))*cosmo.DA(0,z_s)/(cosmo.DA(0,z_d)*cosmo.DA(z_d,z_s))


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

def equatorial_to_polar(RA,Dec,RA_center,Dec_center): 
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
    
    Theta = Theta+np.pi/2
    return Sep, Theta
    
def cap_area(radius):
    """Calculates the solid angle of a cap"""
    Omega = 2.0*np.pi*(1-np.cos(radius))*3282.810874*3600 #arcmin^2
    return Omega
def annular_area(rad1,rad2):
    """Calculates the solid angle of a annulus"""
    return cap_area(rad2)-cap_area(rad1)

def cluster_lensing(cluster,sources, radius):
    """
    Gets the lensing signal given a cluster and a source galaxy catalogue within a 
    radius from which we will get galaxies to measure tangential and cross 
    ellipticities, and the critical lensing density.
    
    cluster = a 3-tuple with cluster RA, DEC (in radians) and redshift.
    sources
    
    sources = a 7-tuple with:
    
    - RA, DEC of galaxy in radians
    - galaxy redshift
    - E1, E2: galaxy ellipticity measurements in the CFHT coordinate system.
    - W: Ellipticity measurement weights, given by 1/(intrinsic_ellipticity^2 + ellipticity_uncertainty^2)
    - M: Estimative for the multiplicative bias of the ellipticity measurement.
    """
    
    
    
    cl_RA, cl_DEC, cl_z = cluster
    cl = np.array([cl_RA,cl_DEC,cl_z]).T

    sr_RA, sr_DEC, sr_z, sr_E1, sr_E2, sr_W, sr_M = sources
    sr=np.array([sr_RA,sr_DEC,sr_z,sr_E1,sr_E2, sr_W,sr_M]).T
    
    cl_DA = cosmo.DA(0,cl_z)
    ang_radius= radius/cl_DA
    
    region_mask= angular_separation(cl_RA,cl_DEC,sr[:,0],sr[:,1])<  ang_radius
    region=sr[region_mask]
    
    bkg_condition = region[:,2]>1.1*cl_z +0.1
    bkg_region = region[bkg_condition,:]
    
    sigs = sigmacrit(cl_z,bkg_region[:,2])
    rads, theta = equatorial_to_polar(bkg_region[:,0],
                    bkg_region[:,1],
                    cl_RA,
                    cl_DEC)

    
    et = -bkg_region[:,3]*np.cos(2*theta) - bkg_region[:,4]*np.sin(2*theta)
    ex = -bkg_region[:,3]*np.sin(2*theta) + bkg_region[:,4]*np.cos(2*theta)
    
    
    result = {'Critical Density': sigs,
              'Tangential Shear': et,
              'Cross Shear': ex,
              'Radial Distance' : rads*cl_DA,
              'Polar Angle': theta,
              'Weights': bkg_region[:,5],
              'Mult. Bias': bkg_region[:,6],
              'Count': len(bkg_region)

        
    }
    
    return result

def stack(clusterbkgs,bins_lims,Nboot=200):
    """
    clusterbkgs = a list of ndarrays, each containing 
    cluster background galaxies for lensing. They should contain: 
    - Sigma_crit the critical density calculated from the cluster and galaxy redshifts
    - e_t the tangential component of the shear
    - e_x the cross component of the shear
    - W the weights of the e_t and e_x measurements
    - M the estimation of multiplicative biases 
    - R the angular diameter radius in Mpc/h between the cluster centre and the background
    galaxy position.
    
    bins_lims = an array containing the bin lower and upper bounds
    
    Nboot = the number of resamplings desired
    """

    resample=np.random.randint(0,len(clusterbkgs),(Nboot,len(clusterbkgs)))
    N=len(bins_lims)
    Delta_Sigmas = np.empty((Nboot,N))
    Delta_Xigmas = np.empty((Nboot,N))
    for sampleNo in range(len(resample)):
        stake = np.hstack([clusterbkgs[i] for i in resample[sampleNo]])

        for radius in range(N):
            bin_max  = stake[:,stake[4,:]<bins_lims[radius,1]]

            bin_min= bin_max[:,bin_max[4,:]>bins_lims[radius,0]]

            Sigma = np.average(bin_min[0,:]*bin_min[1,:],weights= bin_min[3,:]/(bin_min[0,:]**2))
            Xigma = np.average(bin_min[0,:]*bin_min[2,:],weights= bin_min[3,:]/(bin_min[0,:]**2))
            One_plus_K = np.average(bin_min[5,:]+1,weights= bin_min[3,:]/(bin_min[0,:]**2))
            
            Delta_Sigmas[sampleNo,radius] = Sigma/One_plus_K
            Delta_Xigmas[sampleNo,radius] = Xigma/One_plus_K

        del stake


    sigmas_cov=np.cov(Delta_Sigmas.T)
    sigmas = np.mean(Delta_Sigmas,axis=0)
    xigmas = np.mean(Delta_Xigmas,axis=0)
    xigmas_cov = np.cov(Delta_Xigmas.T)
    

    return sigmas, sigmas_cov, xigmas, xigmas_cov