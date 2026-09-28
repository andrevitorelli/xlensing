import numpy as np
import pytest
from xlensing.sims import (
    project_ellipsoid,
    triaxial_NFW_wcs_cluster_shear,
    apply_triaxial_NFW_shear_region,
    NFW_wcs_cluster_shear,
    make_simple_random_cat,
)
from xlensing.cosmo import DA


# ---------------------------------------------------------------------------
# project_ellipsoid
# ---------------------------------------------------------------------------

def test_project_ellipsoid_c_axis_gives_ab_plane():
    # LOS along c-axis (theta=0): projected ellipse is the a-b face
    q_maj, q_min, pa = project_ellipsoid(2.0, 1.5, 0.8, 0.0, 0.0)
    assert np.isclose(q_maj, 2.0, rtol=1e-10)
    assert np.isclose(q_min, 1.5, rtol=1e-10)


def test_project_ellipsoid_a_axis_gives_bc_plane():
    # LOS along a-axis (theta=pi/2, phi=0): projected ellipse is the b-c face
    q_maj, q_min, pa = project_ellipsoid(2.0, 1.5, 0.8, np.pi / 2, 0.0)
    assert np.isclose(q_maj, 1.5, rtol=1e-10)
    assert np.isclose(q_min, 0.8, rtol=1e-10)


def test_project_ellipsoid_b_axis_gives_ac_plane():
    # LOS along b-axis (theta=pi/2, phi=pi/2): projected ellipse is the a-c face
    q_maj, q_min, pa = project_ellipsoid(2.0, 1.5, 0.8, np.pi / 2, np.pi / 2)
    assert np.isclose(q_maj, 2.0, rtol=1e-10)
    assert np.isclose(q_min, 0.8, rtol=1e-10)


def test_project_ellipsoid_sphere_is_circular():
    # A sphere projects to a circle for any LOS
    for theta, phi in [(0.0, 0.0), (0.3, 1.2), (np.pi / 3, 2.0)]:
        q_maj, q_min, pa = project_ellipsoid(1.0, 1.0, 1.0, theta, phi)
        assert np.isclose(q_maj, 1.0, rtol=1e-10)
        assert np.isclose(q_min, 1.0, rtol=1e-10)


def test_project_ellipsoid_axes_ordered():
    # q_maj >= q_min always
    for theta in np.linspace(0, np.pi / 2, 5):
        for phi in np.linspace(0, np.pi, 5):
            q_maj, q_min, _ = project_ellipsoid(3.0, 2.0, 1.0, theta, phi)
            assert q_maj >= q_min - 1e-12


def test_project_ellipsoid_axes_bounded():
    # Projected axes are bounded by the original ellipsoid axes
    a, b, c = 3.0, 2.0, 1.0
    for theta in np.linspace(0, np.pi / 2, 6):
        for phi in np.linspace(0, np.pi, 6):
            q_maj, q_min, _ = project_ellipsoid(a, b, c, theta, phi)
            assert q_maj <= a + 1e-10
            assert q_min >= c - 1e-10


def test_project_ellipsoid_pa_is_scalar():
    q_maj, q_min, pa = project_ellipsoid(2.0, 1.5, 0.8, 0.2, 0.5)
    assert np.isscalar(pa) or pa.ndim == 0


# ---------------------------------------------------------------------------
# triaxial_NFW_wcs_cluster_shear — spherical limit
# ---------------------------------------------------------------------------

M, CONC, ZL, ZS = 1e14, 5.0, 0.3, 1.0
CRA, CDEC = 0.0, 0.0
SRA, SDEC = 0.005, 0.003


def _spherical_q():
    """q_maj = q_min = physical separation at zlens."""
    R = DA(0, ZL) * np.hypot(SRA, SDEC)
    return R, R


def test_triaxial_spherical_limit_matches_spherical():
    q_maj, q_min = _spherical_q()
    g_sph = NFW_wcs_cluster_shear(M, CONC, ZL, ZS, CRA, CDEC, SRA, SDEC)
    g_tri = triaxial_NFW_wcs_cluster_shear(
        M, CONC, ZL, ZS, CRA, CDEC, SRA, SDEC, q_maj, q_min, 0.0
    )
    assert np.isclose(g_sph, g_tri, rtol=1e-4)


def test_triaxial_shear_is_complex():
    q_maj, q_min = _spherical_q()
    g = triaxial_NFW_wcs_cluster_shear(
        M, CONC, ZL, ZS, CRA, CDEC, SRA, SDEC, q_maj, q_min, 0.0
    )
    assert np.iscomplex(g)


def test_triaxial_shear_nonzero_for_background_source():
    q_maj, q_min = _spherical_q()
    g = triaxial_NFW_wcs_cluster_shear(
        M, CONC, ZL, ZS, CRA, CDEC, SRA, SDEC, q_maj, q_min, 0.0
    )
    assert abs(g) > 0


def test_triaxial_stronger_along_major_axis():
    # For an elongated projected ellipse, sources along the major axis
    # experience a smaller effective radius → stronger shear
    sep = 0.004
    q_maj, q_min = 2.0, 1.0   # physical Mpc; elongated north–south (pa=0)
    g_major = triaxial_NFW_wcs_cluster_shear(
        M, CONC, ZL, ZS, CRA, CDEC, 0., sep, q_maj, q_min, 0.0
    )
    g_minor = triaxial_NFW_wcs_cluster_shear(
        M, CONC, ZL, ZS, CRA, CDEC, sep, 0., q_maj, q_min, 0.0
    )
    assert abs(g_major) > abs(g_minor)


def test_triaxial_pa_rotation_changes_asymmetry():
    # Rotating pa by 90° swaps major/minor axis alignment with north/east
    sep = 0.004
    q_maj, q_min = 2.0, 1.0
    g_north_pa0 = triaxial_NFW_wcs_cluster_shear(
        M, CONC, ZL, ZS, CRA, CDEC, 0., sep, q_maj, q_min, 0.0
    )
    g_east_pa90 = triaxial_NFW_wcs_cluster_shear(
        M, CONC, ZL, ZS, CRA, CDEC, sep, 0., q_maj, q_min, np.pi / 2
    )
    # Both are now along their respective major axes → should be equal
    assert np.isclose(abs(g_north_pa0), abs(g_east_pa90), rtol=1e-6)


# ---------------------------------------------------------------------------
# apply_triaxial_NFW_shear_region
# ---------------------------------------------------------------------------

def _make_galaxies(seed=0):
    cat = make_simple_random_cat(
        density=1, width_rad=0.005, zrange=(0.5, 2.0), shape_noise=0.0, seed=seed
    ).T
    return cat.copy()


def test_apply_triaxial_returns_same_shape():
    galaxies = _make_galaxies()
    cluster = [0., 0., 0.3, 1e14, 5.0, 2.0, 1.5, 0.8, 0.0, 0.0]
    result = apply_triaxial_NFW_shear_region(cluster, galaxies.copy())
    assert result.shape == galaxies.shape


def test_apply_triaxial_modifies_ellipticities():
    galaxies = _make_galaxies()
    cluster = [0., 0., 0.3, 1e14, 5.0, 2.0, 1.5, 0.8, 0.0, 0.0]
    original_e = galaxies[:, 3] + 1j * galaxies[:, 4]
    result = apply_triaxial_NFW_shear_region(cluster, galaxies.copy())
    sheared_e = result[:, 3] + 1j * result[:, 4]
    assert not np.allclose(original_e, sheared_e)


def test_apply_triaxial_foreground_unchanged():
    # Galaxies in front of the cluster should not be sheared
    galaxies = _make_galaxies()
    cluster = [0., 0., 0.3, 1e14, 5.0, 2.0, 1.5, 0.8, 0.0, 0.0]
    foreground_mask = galaxies[:, 2] <= cluster[2]
    e_before = (galaxies[foreground_mask, 3] + 1j * galaxies[foreground_mask, 4]).copy()
    result = apply_triaxial_NFW_shear_region(cluster, galaxies.copy())
    e_after = result[foreground_mask, 3] + 1j * result[foreground_mask, 4]
    np.testing.assert_array_equal(e_before, e_after)


def test_apply_triaxial_spherical_limit():
    # When a==b==c, apply_triaxial should give the same ellipticities as
    # apply_NFW_shear_region (spherical).  We compare directly to the triaxial
    # shear formula at q_maj=q_min (spherical equivalence already tested above).
    from xlensing.sims import apply_NFW_shear_region
    galaxies = _make_galaxies(seed=7)
    R = DA(0, 0.3) * 0.01   # characteristic scale

    cluster_sph = [0., 0., 0.3, 1e14, 5.0]
    cluster_tri = [0., 0., 0.3, 1e14, 5.0, R, R, R, 0.0, 0.0]

    g_sph = apply_NFW_shear_region(cluster_sph, galaxies.copy())
    # pseudo path: analytic, so exact agreement
    g_tri = apply_triaxial_NFW_shear_region(cluster_tri, galaxies.copy(), model="pseudo")
    np.testing.assert_allclose(
        g_sph[:, 3] + 1j * g_sph[:, 4],
        g_tri[:, 3] + 1j * g_tri[:, 4],
        rtol=1e-4,
    )
    # exact path: tabulated shear, agreement to the table's ~5e-4 precision
    g_ex = apply_triaxial_NFW_shear_region(cluster_tri, galaxies.copy())
    e_s, e_x = g_sph[:, 3] + 1j * g_sph[:, 4], g_ex[:, 3] + 1j * g_ex[:, 4]
    assert np.max(np.abs(e_x - e_s)) / np.max(np.abs(e_s)) < 2e-3


def test_apply_triaxial_with_pa_offset():
    galaxies = _make_galaxies()
    cluster_0   = [0., 0., 0.3, 1e14, 5.0, 2.0, 1.5, 0.8, 0.0, 0.0, 0.0]
    cluster_pi  = [0., 0., 0.3, 1e14, 5.0, 2.0, 1.5, 0.8, 0.0, 0.0, np.pi]
    result_0  = apply_triaxial_NFW_shear_region(cluster_0,  galaxies.copy())
    result_pi = apply_triaxial_NFW_shear_region(cluster_pi, galaxies.copy())
    # pa and pa+pi give the same projected ellipse (180° symmetry)
    np.testing.assert_allclose(
        result_0[:, 3:5], result_pi[:, 3:5], rtol=1e-10
    )


# --- exact triaxial-NFW shear (xlensing.triaxial, 2026-09-28) -------------------

def test_exact_triaxial_round_limit_matches_spherical():
    """b = c = a must reproduce the spherical shear (old pseudo path) to table precision."""
    import numpy as np
    from xlensing.sims import apply_triaxial_NFW_shear_region, make_simple_random_cat, r_vir
    gal = make_simple_random_cat(30, 0.003, (0.8, 2.0), 0.0, seed=4).T
    z, M, c = 0.4, 2e14, 4.0
    a = r_vir(z, M)
    cl = [0, 0, z, M, c, a, a, a, 0.3, 0.7]
    old = apply_triaxial_NFW_shear_region(cl, gal.copy(), model="pseudo")
    new = apply_triaxial_NFW_shear_region(cl, gal.copy())
    bg = gal[:, 2] > z
    e_o, e_n = old[bg, 3] + 1j * old[bg, 4], new[bg, 3] + 1j * new[bg, 4]
    assert np.max(np.abs(e_n - e_o)) / np.max(np.abs(e_o)) < 2e-3


def test_exact_triaxial_circle_identity():
    """Azimuthal mean tangential shear = kbar(<R) - k(R) for any mass distribution."""
    import numpy as np
    import xlensing.triaxial as T
    q = 0.5
    for R in (0.2, 1.0, 3.0):
        th = np.linspace(0, 2 * np.pi, 361)[:-1]
        x, y = R * np.cos(th), R * np.sin(th)
        g1, g2, _ = T._shear_grid(q, x, y)
        gt = np.mean(-(g1 * np.cos(2 * th) + g2 * np.sin(2 * th)))
        rr = np.linspace(0, R, 401)[1:] - R / 800
        RR, TT = np.meshgrid(rr, np.linspace(0, 2 * np.pi, 181)[:-1])
        xi = np.sqrt((RR * np.cos(TT)) ** 2 + (RR * np.sin(TT)) ** 2 / q ** 2)
        kin = np.sum(T._sig(xi) * RR) * (R / 400) * (2 * np.pi / 180) / (np.pi * R ** 2)
        kring = np.mean(T._sig(np.sqrt(x ** 2 + y ** 2 / q ** 2)))
        assert abs(gt / (kin - kring) - 1) < 2e-3


def test_exact_triaxial_projection_conserves_mass():
    """det(M) * n^T Q n = det(Q) = 1: the projected mass is orientation independent."""
    import numpy as np
    from xlensing.triaxial import projected_ellipse
    rng = np.random.default_rng(0)
    for _ in range(20):
        axes = np.sort(rng.uniform(0.3, 1.0, 3))[::-1]
        lam1, lam2, _, nQn = projected_ellipse(axes, np.arccos(rng.uniform()), rng.uniform(0, 2 * np.pi))
        assert abs(lam1 * lam2 * nQn - 1) < 1e-10
