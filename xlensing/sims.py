"""Module for simulating/injecting simulated clusters on galaxy catalogs."""

import numpy as np
import ngmix
from xlensing.cosmo import DA, cosmology,cluster_overdensity,lightspeed,gravity,OmegaM


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

  if shape_noise != 0:
      sources_E1, sources_E2 = ngmix.priors.shape.GPriorBA(sigma=shape_noise,rng=rng).sample2d(Ngals)
  else:
      sources_E1, sources_E2 = np.zeros(Ngals), np.zeros(Ngals)
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

def project_ellipsoid(a, b, c, theta_los, phi_los):
    """
    Project a triaxial ellipsoid onto the plane of the sky.

    Given a 3D ellipsoid with semi-axes (a, b, c) aligned with its principal
    axes, and a line-of-sight direction expressed in that same frame, compute
    the projected 2D ellipse via the Schur complement of the ellipsoid's
    shape matrix.

    Parameters
    ----------
    a, b, c : float
        Semi-axes of the ellipsoid (a >= b >= c), in any consistent length
        unit (typically Mpc).
    theta_los : float
        Polar angle of the line of sight in the ellipsoid principal frame [rad].
        theta_los=0 means the LOS is along the c-axis (most face-on).
    phi_los : float
        Azimuthal angle of the LOS in the ellipsoid principal frame [rad].
        phi_los=0 means the LOS projected onto the ab-plane is along the a-axis.

    Returns
    -------
    q_maj : float
        Projected semi-major axis (same units as a, b, c).
    q_min : float
        Projected semi-minor axis.
    pa : float
        Position angle of the projected major axis [rad], measured
        counterclockwise from the projection of the a-axis onto the sky plane.
        Add a constant offset to convert to the RA/Dec frame if needed.
    """
    sth, cth = np.sin(theta_los), np.cos(theta_los)
    sph, cph = np.sin(phi_los),   np.cos(phi_los)
    n = np.array([sth * cph, sth * sph, cth])   # LOS unit vector

    Q = np.diag([1.0 / a**2, 1.0 / b**2, 1.0 / c**2])  # ellipsoid shape matrix

    # Sky-plane basis: e1 = projection of a-axis (1,0,0) onto the sky plane
    e1 = np.array([1.0, 0.0, 0.0]) - np.dot([1.0, 0.0, 0.0], n) * n
    if np.linalg.norm(e1) < 1e-10:          # LOS nearly along a-axis; fall back to b-axis
        e1 = np.array([0.0, 1.0, 0.0]) - np.dot([0.0, 1.0, 0.0], n) * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)                    # completes right-handed sky frame

    # 2x2 projected ellipse matrix (Schur complement of Q along LOS)
    Qn  = Q @ n
    nQn = float(n @ Qn)
    M = np.array([
        [e1 @ Q @ e1 - (e1 @ Qn)**2 / nQn,
         e1 @ Q @ e2 - (e1 @ Qn) * (e2 @ Qn) / nQn],
        [e2 @ Q @ e1 - (e2 @ Qn) * (e1 @ Qn) / nQn,
         e2 @ Q @ e2 - (e2 @ Qn)**2 / nQn],
    ])

    # Eigendecomposition: eigenvalues ascending → smaller eigval = larger axis
    eigvals, eigvecs = np.linalg.eigh(M)
    q_maj = 1.0 / np.sqrt(eigvals[0])        # major semi-axis
    q_min = 1.0 / np.sqrt(eigvals[1])        # minor semi-axis
    pa    = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])  # PA of major axis from e1

    return q_maj, q_min, pa


def triaxial_NFW_wcs_cluster_shear(
    Mlens, conc, zlens, zsource,
    cluster_RA, cluster_DEC, source_RA, source_DEC,
    q_maj, q_min, pa,
):
    """
    NFW shear for an elliptically projected mass distribution.

    Uses a spherical NFW profile evaluated at an effective radius obtained
    by affine-rescaling the projected position through the ellipse
    (q_maj, q_min, pa).  The shear direction is derived from the gradient of
    the elliptic-radius field so that it reduces exactly to the tangential
    convention of NFW_wcs_cluster_shear when q_maj == q_min.

    Parameters
    ----------
    Mlens : float
        Cluster mass M200 [M_sun].
    conc : float
        NFW concentration c200.
    zlens, zsource : float
        Cluster and source redshifts.
    cluster_RA, cluster_DEC : float
        Cluster sky coordinates [rad].
    source_RA, source_DEC : float
        Source sky coordinates [rad].
    q_maj, q_min : float
        Projected semi-major / semi-minor axes [Mpc].
    pa : float
        Position angle of the projected major axis [rad], measured from north
        (Dec axis) toward east (RA axis) — standard astronomical convention.

    Returns
    -------
    complex
        Complex shear g = g1 + i*g2 in the RA/Dec frame.
    """
    cluster_DA = DA(0, zlens)
    sep, theta = equatorial_to_polar(cluster_RA, cluster_DEC, source_RA, source_DEC)

    # Physical 2D offset at the lens plane (Mpc), east (+x) and north (+y)
    dx = cluster_DA * sep * np.sin(theta)
    dy = cluster_DA * sep * np.cos(theta)

    # Rotate to projected ellipse principal frame (u along major axis, v along minor)
    # The rotation matrix is symmetric and involutory for this PA convention
    u =  dx * np.sin(pa) + dy * np.cos(pa)
    v =  dx * np.cos(pa) - dy * np.sin(pa)

    # Effective circularised radius; preserves area of isodensity contours
    r_eff = np.sqrt((u / q_maj)**2 + (v / q_min)**2) * np.sqrt(q_maj * q_min)

    # Gradient of the elliptic-radius field, rotated back to the (east, north) frame
    # Reduces to (dx/R, dy/R) for the spherical case, giving phi_eff = theta
    gx = np.sin(pa) * u / q_maj**2 + np.cos(pa) * v / q_min**2   # east component
    gy = np.cos(pa) * u / q_maj**2 - np.sin(pa) * v / q_min**2   # north component
    phi_eff = np.arctan2(gx, gy)    # PA from north, same convention as equatorial_to_polar

    gammat = NFW_tangential_shear(Mlens, conc, zlens, zsource, r_eff)
    return gammat * np.exp(2j * phi_eff)


def apply_triaxial_NFW_shear_region(cluster, galaxies):
    """Apply a triaxial NFW shear over a catalog of galaxies.

    The 3D triaxial ellipsoid is projected onto the sky via
    project_ellipsoid, and the resulting 2D ellipse is used to remap each
    galaxy's projected separation to an effective spherical radius before
    evaluating the NFW profile.

    Parameters
    ----------
    cluster : array-like, length 10 or 11
        [RA, DEC, z, M200, C200, a, b, c, theta_los, phi_los] and optionally
        [pa_sky_offset].
        - RA, DEC: cluster centre [rad]
        - z: cluster redshift
        - M200: mass [M_sun]
        - C200: NFW concentration
        - a, b, c: 3D semi-axes of the ellipsoid [Mpc], a >= b >= c
        - theta_los, phi_los: LOS direction in the ellipsoid principal frame [rad]
        - pa_sky_offset: optional extra rotation [rad] applied to the projected
          PA to align the ellipsoid frame with RA/Dec (default 0).
    galaxies : ndarray (Ngal, >=5)
        Galaxy catalog rows: RA, DEC, z, e1, e2, ...

    Returns
    -------
    ndarray
        galaxies array with columns 3 (e1) and 4 (e2) updated.
    """
    a, b, c           = cluster[5], cluster[6], cluster[7]
    theta_los, phi_los = cluster[8], cluster[9]
    pa_offset          = cluster[10] if len(cluster) > 10 else 0.0

    q_maj, q_min, pa_intrinsic = project_ellipsoid(a, b, c, theta_los, phi_los)
    pa = pa_intrinsic + pa_offset

    gammas = []
    for galaxy in galaxies:
        if galaxy[2] > cluster[2]:
            gammas.append(triaxial_NFW_wcs_cluster_shear(
                cluster[3], cluster[4], cluster[2], galaxy[2],
                cluster[0], cluster[1], galaxy[0], galaxy[1],
                q_maj, q_min, pa,
            ))
        else:
            gammas.append(0)

    gammas    = np.array(gammas)
    sheared_e = add_shears(galaxies[:, 3] + 1j * galaxies[:, 4], gammas)
    galaxies[:, 3] = sheared_e.real
    galaxies[:, 4] = sheared_e.imag
    return galaxies


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