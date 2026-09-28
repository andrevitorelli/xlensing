"""Exact weak-lensing shear of a projected triaxial NFW halo.

Physics (e.g. Oguri, Lee & Suto 2003; Corless & King 2007)
----------------------------------------------------------
The 3D density is NFW in the ellipsoidal radius

    m^2 = x^2/alpha^2 + y^2/beta^2 + z^2/gamma^2,   alpha*beta*gamma = 1,

with (x, y, z) in the halo's principal frame.  The unit-determinant
normalisation keeps the volume inside every isodensity surface equal to that
of the sphere of the same radius, so the enclosed mass -- and therefore M200
and c200 -- keep their spherical meaning.

Projecting along a line of sight n, with Q = diag(alpha^-2, beta^-2,
gamma^-2) and M the 2x2 Schur complement of Q along n (the projected-ellipse
matrix), the surface density is exactly elliptical:

    Sigma(x_perp) = Sigma_sph(xi) / sqrt(n^T Q n),     xi^2 = x_perp^T M x_perp,

where Sigma_sph is the spherical NFW surface density.  The 1/sqrt(n^T Q n)
factor is the line-of-sight projection amplitude (larger when the halo is
elongated along the line of sight); since det M * n^T Q n = det Q = 1, the
total projected mass is the same for every orientation.

In the frame of the projected major (u) and minor (v) axes, with M's
eigenvalues lam1 <= lam2, xi^2 = lam1 * (u^2 + v^2/q^2), q = sqrt(lam1/lam2).
The shear of an elliptical convergence kappa(xi_K), xi_K^2 = x^2 + y^2/q^2, is
given by one-dimensional integrals (Schramm 1990; Keeton 2001, astro-ph/0102341):

    psi_xx = 2 q x^2 K0 + q J0,  psi_yy = 2 q y^2 K2 + q J1,  psi_xy = 2 q x y K1,
    J_n = int_0^1 kappa(xi(u)^2) / [1-(1-q^2)u]^(n+1/2) du,
    K_n = int_0^1 u kappa'(xi(u)^2) / [1-(1-q^2)u]^(n+1/2) du,
    xi(u)^2 = u [x^2 + y^2 / (1-(1-q^2)u)],   kappa' = d kappa / d(xi^2),

gamma_1 = (psi_xx - psi_yy)/2, gamma_2 = psi_xy.  For the NFW the profile has
the same shape at every mass once lengths are measured in units of the scale
radius, so the dimensionless shear of kappa = SigX(xi_K) (SigX: NFW surface
density in units of 2 r_s delta_c rho_m) is tabulated once on (log r, angle,
q) and interpolated per galaxy.
"""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline, RegularGridInterpolator

from xlensing.model import SigX

_TABLE_PATH = os.path.join(os.path.dirname(__file__), "triaxial_nfw_shear_table.npz")
_LOGR = np.linspace(-3.5, 3.0, 261)          # log10 of radius in units of the major-axis scale
_PHI = np.linspace(0.0, np.pi / 2, 49)      # angle from the major axis (one quadrant)
_Q = np.linspace(0.10, 1.00, 181)           # projected minor/major axis ratio


# --- dimensionless NFW surface density and its derivative -------------------------
_lx = np.linspace(-8.0, 6.0, 4001)
_lg = np.log(SigX(10.0 ** _lx))
_spl = CubicSpline(_lx * np.log(10.0), _lg)            # ln SigX as a function of ln x


def _sig(x):
    return np.exp(_spl(np.log(np.clip(x, 1e-8, 1e6))))


def _dsig_dxi2(x):
    """d SigX / d(x^2) = SigX'(x) / (2x) = SigX * (d ln SigX/d ln x) / (2 x^2)."""
    xc = np.clip(x, 1e-8, 1e6)
    return _sig(xc) * _spl(np.log(xc), 1) / (2.0 * xc ** 2)


def _gt_round(r):
    """Dimensionless tangential shear of the round NFW, SigBar - SigX, via the spline."""
    from xlensing.model import SigBar
    r = np.clip(np.asarray(r, dtype=float), 1e-8, 1e6)
    return SigBar(r) - _sig(r)


def _shear_grid(q, x, y, n_nodes=256):
    """Dimensionless (gamma1, gamma2, kappa) of kappa = SigX(xi), xi^2 = x^2 + y^2/q^2,
    at points (x, y) in the major/minor-axis frame (x along the major axis)."""
    t, wt = np.polynomial.legendre.leggauss(n_nodes)
    t = 0.5 * (t + 1.0); wt = 0.5 * wt
    u = t ** 2; wu = 2.0 * t * wt                          # u = t^2 handles the centre
    x = np.asarray(x)[..., None]; y = np.asarray(y)[..., None]
    d = 1.0 - (1.0 - q ** 2) * u
    xi2 = u * (x ** 2 + y ** 2 / d)
    xi = np.sqrt(xi2)
    kap = _sig(xi); kp = _dsig_dxi2(xi)
    J0 = np.sum(wu * kap / d ** 0.5, -1); J1 = np.sum(wu * kap / d ** 1.5, -1)
    K0 = np.sum(wu * u * kp / d ** 0.5, -1); K1 = np.sum(wu * u * kp / d ** 1.5, -1)
    K2 = np.sum(wu * u * kp / d ** 2.5, -1)
    x = x[..., 0]; y = y[..., 0]
    pxx = 2 * q * x ** 2 * K0 + q * J0
    pyy = 2 * q * y ** 2 * K2 + q * J1
    pxy = 2 * q * x * y * K1
    return 0.5 * (pxx - pyy), pxy, 0.5 * (pxx + pyy)


def build_table(path=_TABLE_PATH, n_nodes=256):
    """Tabulate the dimensionless shear on (log r, phi, q); ~1 minute, run once."""
    R, P = np.meshgrid(10.0 ** _LOGR, _PHI, indexing="ij")
    X, Y = R * np.cos(P), R * np.sin(P)
    g1 = np.empty((len(_LOGR), len(_PHI), len(_Q))); g2 = np.empty_like(g1); kp = np.empty_like(g1)
    for k, q in enumerate(_Q):
        g1[..., k], g2[..., k], kp[..., k] = _shear_grid(q, X, Y, n_nodes)
    ref = _gt_round(R)[..., None]
    xi = np.sqrt(X[..., None] ** 2 + Y[..., None] ** 2 / _Q ** 2)
    # tabulate smooth ratios: shear / round tangential shear at the same r, kappa / SigX(xi)
    np.savez(path, logr=_LOGR, phi=_PHI, q=_Q, g1=g1 / ref, g2=g2 / ref, kappa=kp / _sig(xi))
    return path


@lru_cache(maxsize=1)
def _interpolators():
    if not os.path.exists(_TABLE_PATH):
        build_table()
    t = np.load(_TABLE_PATH)
    grid = (t["logr"], t["phi"], t["q"])
    mk = lambda a: RegularGridInterpolator(grid, a, bounds_error=False, fill_value=None)
    return mk(t["g1"]), mk(t["g2"]), mk(t["kappa"])


def projected_ellipse(axes, theta_los, phi_los):
    """Projection of a triaxial halo with semi-axes proportional to ``axes`` (a >= b >= c).

    Returns (lam1, lam2, pa, nQn) for the unit-determinant (volume-preserving)
    shape matrix: M's eigenvalues lam1 <= lam2 (xi^2 = lam1 u^2 + lam2 v^2),
    the major-axis position angle pa in the same convention as
    ``xlensing.sims.project_ellipsoid``, and n^T Q n.
    """
    a, b, c = np.asarray(axes, dtype=float) / np.prod(axes) ** (1.0 / 3.0)
    sth, cth = np.sin(theta_los), np.cos(theta_los)
    sph, cph = np.sin(phi_los), np.cos(phi_los)
    n = np.array([sth * cph, sth * sph, cth])
    Q = np.diag([1.0 / a ** 2, 1.0 / b ** 2, 1.0 / c ** 2])
    e1 = np.array([1.0, 0.0, 0.0]) - n[0] * n
    if np.linalg.norm(e1) < 1e-10:
        e1 = np.array([0.0, 1.0, 0.0]) - n[1] * n
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n, e1)
    Qn = Q @ n; nQn = float(n @ Qn)
    E = np.column_stack([e1, e2])
    M = E.T @ Q @ E - np.outer(E.T @ Qn, E.T @ Qn) / nQn
    w, v = np.linalg.eigh(M)
    pa = np.arctan2(v[1, 0], v[0, 0])
    return w[0], w[1], pa, nQn


def triaxial_nfw_shear(u, v, rs, lam1, lam2, nQn):
    """Dimensionless shear and convergence of a projected triaxial NFW.

    ``u``, ``v``: galaxy coordinates [Mpc] along the projected major and minor
    axes; ``rs``: scale radius [Mpc].  Returns (gamma1, gamma2, kappa) in the
    (u, v) frame, in units of 2 r_s delta_c rho_m / Sigma_crit (multiply by it
    to get the shear).  Includes the line-of-sight amplitude 1/sqrt(nQn).
    """
    q = np.sqrt(lam1 / lam2)
    ell = rs / np.sqrt(lam1)                 # major-axis scale length
    x, y = np.asarray(u) / ell, np.asarray(v) / ell
    r = np.hypot(x, y)
    phi = np.arctan2(np.abs(y), np.abs(x))
    pts = np.column_stack([np.log10(np.clip(r, 10 ** _LOGR[0], 10 ** _LOGR[-1])), phi, np.full(r.shape, q)])
    i1, i2, ik = _interpolators()
    amp = 1.0 / np.sqrt(nQn)
    ref = _gt_round(np.clip(r, 10 ** _LOGR[0], 10 ** _LOGR[-1]))
    g1 = i1(pts) * ref; g2 = i2(pts) * np.sign(x * y) * ref
    kap = ik(pts) * _sig(np.sqrt(x ** 2 + y ** 2 / q ** 2))
    return amp * g1, amp * g2, amp * kap
