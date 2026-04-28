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
    result = {
        'sep': sep,
        'theta': theta
    }
    return result 
    
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
      
      - Critical density: the \\Sigma_{crit}, for weak lensing (N values)
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
    polar_background = equatorial_to_polar(cluster[0],cluster[1],source[:,0],source[:,1])
    region_mask = polar_background['sep'] <  angular_radius
    region = source[region_mask]
    
    #select galaxy backgrounds
    background_condition = (region[:,2]> 1.1*cluster[2] +.1)   #this is contentious and should be changed
    background_region = region[background_condition,:]

    #critical lensing density and polar position of sources/clusters
    sigs = sigmacrit(cluster[2],background_region[:,2])/1e12 #msun/pc^2 is better than msun/mpc^2 for numerical reasons
    polar_region= equatorial_to_polar(background_region[:,0],
                    background_region[:,1],
                    cluster[0],
                    cluster[1])
    rads, theta = polar_region['sep'], polar_region['theta']
    theta += sys_angle
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
      
      - Critical density: the \\Sigma_{crit}, for weak lensing (N values)
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
    region_mask = equatorial_to_polar(cluster[0],cluster[1],source[:,0],source[:,1])['sep']<  angular_radius
    region = source[region_mask]
    
    #select galaxy backgrounds
    background_condition = (region[:,2]> 1.1*cluster[2] +0.2)   #this is contentious and should be changed
    background_region = region[background_condition,:]

    #critical lensing density and polar position of sources/clusters
    sigs = sigmacrit(cluster[2],background_region[:,2])/1e12 #msun/pc^2 is better than msun/mpc^2 for numerical reasons
    polar_region = equatorial_to_polar(background_region[:,0],
                    background_region[:,1],
                    cluster[0],
                    cluster[1])
    rads, theta = polar_region['sep'], polar_region['theta']
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

def signal(stake, bin_limits):
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
  bin_limits = np.asarray(bin_limits)
  sig, et, ex, w, R, M = stake[0], stake[1], stake[2], stake[3], stake[4], stake[5]
  w_eff = w / sig**2

  # in_bin[g, b] is True when galaxy g falls in radial bin b — shape (N_gal, Nbins)
  in_bin = (R[:, None] > bin_limits[:, 0]) & (R[:, None] < bin_limits[:, 1])

  # ΔΣ = Σ(e_t · W/Σ_crit) / Σ((1+M) · W/Σ_crit²)  — the shared Σ(w_eff) cancels
  num_t   = np.dot(et * w / sig,     in_bin)  # (Nbins,)
  num_x   = np.dot(ex * w / sig,     in_bin)
  denom_K = np.dot((1 + M) * w_eff,  in_bin)

  return num_t / denom_K, num_x / denom_K


def stacked_signal(cluster_backgrounds, bin_limits, Nboot=200):
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
    bin_limits = np.asarray(bin_limits)
    N_clusters = len(cluster_backgrounds)
    Nbins = len(bin_limits)

    # --- galaxy counts per bin for diagnostics and boost factors ---
    sources_radii = np.hstack([bg[4] for bg in cluster_backgrounds])
    in_bin_all = (sources_radii[:, None] > bin_limits[:, 0]) & (sources_radii[:, None] < bin_limits[:, 1])
    bin_counts = in_bin_all.sum(axis=0)
    total_gals = bin_counts.sum()
    print("Total galaxies available per bin:")
    print(bin_counts)
    print()

    # --- boost factors ---
    max_radius, min_radius = np.max(bin_limits), np.min(bin_limits)
    area = np.pi * (max_radius**2 - min_radius**2)
    density = total_gals / area
    RR_n = round(density * 4 * max_radius**2)
    RRx = np.random.uniform(-max_radius, max_radius, RR_n)
    RRy = np.random.uniform(-max_radius, max_radius, RR_n)
    RR = np.hypot(RRx, RRy)
    RR_bins = ((RR[:, None] > bin_limits[:, 0]) & (RR[:, None] < bin_limits[:, 1])).sum(axis=0)
    boosts = bin_counts / RR_bins

    # --- precompute per-cluster per-bin weighted sums ---
    # Decompose the signal so that bootstrap resampling is a pure array operation:
    # ΔΣ_boot = Σ_c(sums_t[c]) / Σ_c(sums_K[c])  for resampled cluster indices c
    sums_t = np.zeros((N_clusters, Nbins))
    sums_x = np.zeros((N_clusters, Nbins))
    sums_K = np.zeros((N_clusters, Nbins))

    for c, bg in enumerate(cluster_backgrounds):
        sig, et, ex, w, R, M = bg[0], bg[1], bg[2], bg[3], bg[4], bg[5]
        w_eff = w / sig**2
        in_bin = (R[:, None] > bin_limits[:, 0]) & (R[:, None] < bin_limits[:, 1])  # (N_gal, Nbins)
        sums_t[c] = np.dot(et * w / sig,    in_bin)
        sums_x[c] = np.dot(ex * w / sig,    in_bin)
        sums_K[c] = np.dot((1 + M) * w_eff, in_bin)

    # --- vectorised bootstrap: no Python loop over Nboot ---
    resample = np.random.randint(0, N_clusters, (Nboot, N_clusters))  # (Nboot, N_clusters)
    boot_t = sums_t[resample].sum(axis=1)  # (Nboot, Nbins)
    boot_x = sums_x[resample].sum(axis=1)
    boot_K = sums_K[resample].sum(axis=1)

    Delta_Sigmas = boot_t / boot_K
    Delta_Xigmas = boot_x / boot_K

    sigmas = Delta_Sigmas.mean(axis=0)
    xigmas = Delta_Xigmas.mean(axis=0)
    sigmas_cov = np.cov(Delta_Sigmas.T)
    xigmas_cov = np.cov(Delta_Xigmas.T)

    return sigmas, boosts, sigmas_cov, xigmas, xigmas_cov

def single_cluster(cluster_backgrounds, bin_limits, Nboot=500):
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
    bin_limits = np.asarray(bin_limits)
    Nbins = len(bin_limits)

    sources_radii = np.hstack([bg[4] for bg in cluster_backgrounds])
    in_bin_all = (sources_radii[:, None] > bin_limits[:, 0]) & (sources_radii[:, None] < bin_limits[:, 1])
    print("Total galaxies available per bin:")
    print(in_bin_all.sum(axis=0).tolist())
    print()

    print("Single cluster:")
    background = cluster_backgrounds[0]
    sig, et, ex, w, R, M = background[0], background[1], background[2], background[3], background[4], background[5]

    Delta_Sigmas = np.full((Nboot, Nbins), np.nan)
    Delta_Xigmas = np.full((Nboot, Nbins), np.nan)

    # Vectorise over Nboot within each bin: draw all bootstrap indices at once,
    # then use advanced indexing bin_gals[:, idx] → (6, Nboot, N_gal) to avoid
    # the inner Python loop over Nboot.
    for b, (r_lo, r_hi) in enumerate(bin_limits):
        mask = (R > r_lo) & (R < r_hi)
        N_gal = mask.sum()
        if N_gal == 0:
            continue

        bin_gals = background[:, mask]                          # (6, N_gal)
        idx = np.random.randint(0, N_gal, (Nboot, N_gal))      # (Nboot, N_gal)
        resampled = bin_gals[:, idx]                            # (6, Nboot, N_gal)

        sig_b = resampled[0]   # (Nboot, N_gal)
        et_b  = resampled[1]
        ex_b  = resampled[2]
        w_b   = resampled[3]
        M_b   = resampled[5]

        num_t   = (et_b * w_b / sig_b).sum(axis=-1)           # (Nboot,)
        num_x   = (ex_b * w_b / sig_b).sum(axis=-1)
        denom_K = ((1 + M_b) * w_b / sig_b**2).sum(axis=-1)

        Delta_Sigmas[:, b] = num_t / denom_K
        Delta_Xigmas[:, b] = num_x / denom_K

    sigmas = np.nanmean(Delta_Sigmas, axis=0)
    xigmas = np.nanmean(Delta_Xigmas, axis=0)
    sigmas_cov = np.cov(Delta_Sigmas.T)
    xigmas_cov = np.cov(Delta_Xigmas.T)

    return sigmas, sigmas_cov, xigmas, xigmas_cov


