import numpy as np
import pytest
from xlensing import data as d


# ---------------------------------------------------------------------------
# Helpers / factories
# ---------------------------------------------------------------------------

def _make_sources_lensfit(N, z_cluster=0.3, rng=None):
    """Minimal lensfit source catalog placed just north of the cluster."""
    if rng is None:
        rng = np.random.default_rng(0)
    z_cluster_ra = 0.0
    z_cluster_dec = 0.0
    # place sources slightly north so they pass the angular-radius cut
    ra  = rng.uniform(-0.005, 0.005, N)
    dec = rng.uniform(0.001, 0.005, N)
    z_s = rng.uniform(z_cluster * 1.1 + 0.15, 1.0, N)
    e1  = rng.uniform(-0.3, 0.3, N)
    e2  = rng.uniform(-0.3, 0.3, N)
    W   = rng.uniform(0.5, 2.0, N)
    M   = rng.uniform(-0.05, 0.05, N)
    return (ra, dec, z_s, e1, e2, W, M)


def _make_sources_metacal(N, z_cluster=0.3, rng=None):
    """Minimal metacal source catalog placed near the cluster."""
    if rng is None:
        rng = np.random.default_rng(1)
    ra  = rng.uniform(-0.005, 0.005, N)
    dec = rng.uniform(0.001, 0.005, N)
    z_s = rng.uniform(z_cluster * 1.1 + 0.25, 1.0, N)
    e1  = rng.uniform(-0.3, 0.3, N)
    e2  = rng.uniform(-0.3, 0.3, N)
    W   = rng.uniform(0.5, 2.0, N)
    R11 = rng.uniform(0.5, 0.8, N)
    R12 = rng.uniform(-0.05, 0.05, N)
    R21 = rng.uniform(-0.05, 0.05, N)
    R22 = rng.uniform(0.5, 0.8, N)
    return (ra, dec, z_s, e1, e2, W, R11, R12, R21, R22)


def _make_stake(N=200, rng=None):
    """Return a (6, N) background array with physically reasonable values."""
    if rng is None:
        rng = np.random.default_rng(7)
    sig = rng.uniform(1.0, 5.0, N)          # Σ_crit (M☉/pc²)
    et  = rng.uniform(-0.3, 0.3, N)
    ex  = rng.uniform(-0.3, 0.3, N)
    W   = rng.uniform(0.5, 2.0, N)
    R   = rng.uniform(0.1, 5.0, N)          # radial distance (Mpc)
    M   = rng.uniform(-0.05, 0.05, N)
    return np.vstack([sig, et, ex, W, R, M])


# ---------------------------------------------------------------------------
# equatorial_to_polar
# ---------------------------------------------------------------------------

def test_equatorial_to_polar_returns_dict():
    result = d.equatorial_to_polar(0.0, 0.0, 0.1, 0.1)
    assert isinstance(result, dict)
    assert 'sep' in result and 'theta' in result


def test_equatorial_to_polar_same_point_zero_sep():
    result = d.equatorial_to_polar(0.1, 0.2, 0.1, 0.2)
    np.testing.assert_allclose(result['sep'], 0.0, atol=1e-12)


def test_equatorial_to_polar_separation_nonnegative():
    lon1 = np.random.default_rng(3).uniform(-np.pi, np.pi, 20)
    lat1 = np.random.default_rng(4).uniform(-np.pi/4, np.pi/4, 20)
    result = d.equatorial_to_polar(lon1, lat1, lon1 * 0.9, lat1 * 1.1)
    assert np.all(result['sep'] >= 0)


def test_equatorial_to_polar_known_separation():
    # Two points 0.1 rad apart on the equator
    result = d.equatorial_to_polar(0.0, 0.0, 0.1, 0.0)
    np.testing.assert_allclose(result['sep'], 0.1, rtol=1e-6)


def test_equatorial_to_polar_vectorized():
    N = 50
    rng = np.random.default_rng(5)
    lon = rng.uniform(0, 0.5, N)
    lat = rng.uniform(0, 0.5, N)
    result = d.equatorial_to_polar(0.0, 0.0, lon, lat)
    assert result['sep'].shape == (N,)
    assert result['theta'].shape == (N,)


# ---------------------------------------------------------------------------
# cap_area / annular_area
# ---------------------------------------------------------------------------

def test_cap_area_zero_at_zero():
    np.testing.assert_allclose(d.cap_area(0.0), 0.0, atol=1e-10)


def test_cap_area_positive():
    assert d.cap_area(0.01) > 0


def test_cap_area_increases_with_radius():
    r = np.array([0.001, 0.01, 0.1])
    areas = np.array([d.cap_area(x) for x in r])
    assert np.all(np.diff(areas) > 0)


def test_annular_area_equals_cap_difference():
    r1, r2 = 0.005, 0.02
    np.testing.assert_allclose(
        d.annular_area(r1, r2),
        d.cap_area(r2) - d.cap_area(r1),
        rtol=1e-12,
    )


def test_annular_area_zero_for_equal_radii():
    np.testing.assert_allclose(d.annular_area(0.01, 0.01), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# sigmacrit
# ---------------------------------------------------------------------------

def test_sigmacrit_positive():
    assert d.sigmacrit(0.3, 0.8) > 0


def test_sigmacrit_decreases_with_source_redshift():
    # higher z_s → smaller DA(z_d, z_s) / DA(0, z_s) → larger Σ_crit
    # actually Σ_crit = (c²/4πG) · DA(0,z_s) / (DA(0,z_d)·DA(z_d,z_s))
    # DA(0,z_s) / DA(z_d,z_s) decreases as z_s → z_d from above, then increases
    # For fixed z_d, larger z_s means more lensing efficiency → lower Σ_crit
    sc1 = d.sigmacrit(0.3, 0.6)
    sc2 = d.sigmacrit(0.3, 1.5)
    assert sc2 < sc1


def test_sigmacrit_depends_on_lens_redshift():
    # Different lens redshifts should give different Σ_crit for same source
    sc1 = d.sigmacrit(0.2, 0.8)
    sc2 = d.sigmacrit(0.5, 0.8)
    assert sc1 != sc2


# ---------------------------------------------------------------------------
# tangential_response
# ---------------------------------------------------------------------------

def test_tangential_response_phi_zero():
    # phi=0: Rt = R11·1 + R22·0 + (R12+R21)·0·1 = R11
    np.testing.assert_allclose(d.tangential_response(0.7, 0.01, 0.01, 0.7, 0.0), 0.7)


def test_tangential_response_phi_quarter_pi():
    # phi=π/4: cos²(π/2)=0, sin²(π/2)=1 → Rt = R22
    np.testing.assert_allclose(d.tangential_response(0.7, 0.0, 0.0, 0.8, np.pi/4), 0.8)


def test_tangential_response_identity():
    # R11=R22=1, R12=R21=0 → Rt=1 for any phi
    phi = np.linspace(0, 2*np.pi, 20)
    Rt = d.tangential_response(1.0, 0.0, 0.0, 1.0, phi)
    np.testing.assert_allclose(Rt, 1.0, rtol=1e-12)


def test_tangential_response_vectorized():
    phi = np.linspace(0, np.pi, 30)
    Rt = d.tangential_response(0.7, 0.01, 0.01, 0.7, phi)
    assert Rt.shape == (30,)


# ---------------------------------------------------------------------------
# lensfit_cluster_lensing
# ---------------------------------------------------------------------------

@pytest.fixture
def lensfit_result():
    cluster = (0.0, 0.0, 0.3)
    sources = _make_sources_lensfit(500, z_cluster=0.3, rng=np.random.default_rng(10))
    return d.lensfit_cluster_lensing(cluster, sources, radius=10.0)


def test_lensfit_returns_expected_keys(lensfit_result):
    for key in ['Critical Density', 'Tangential Shear', 'Cross Shear',
                'Radial Distance', 'Polar Angle', 'Weights', 'Mult. Bias', 'Count']:
        assert key in lensfit_result


def test_lensfit_critical_density_positive(lensfit_result):
    assert np.all(lensfit_result['Critical Density'] > 0)


def test_lensfit_count_consistent(lensfit_result):
    N = lensfit_result['Count']
    assert len(lensfit_result['Critical Density']) == N
    assert len(lensfit_result['Tangential Shear']) == N


def test_lensfit_radial_distance_within_radius(lensfit_result):
    # All selected galaxies must be within the requested radius (10 Mpc)
    assert np.all(lensfit_result['Radial Distance'] <= 10.0)


def test_lensfit_radial_distance_positive(lensfit_result):
    assert np.all(lensfit_result['Radial Distance'] > 0)


def test_lensfit_foreground_excluded():
    # Mix foreground + background sources; only background should be counted.
    cluster = (0.0, 0.0, 0.5)
    rng = np.random.default_rng(20)
    N_fg, N_bg = 100, 50
    N = N_fg + N_bg
    ra  = rng.uniform(-0.003, 0.003, N)
    dec = rng.uniform(0.001, 0.003, N)
    # first N_fg are foreground (z < cut), last N_bg are background (z >> cut)
    z_fg = rng.uniform(0.1, 0.4, N_fg)
    z_bg = rng.uniform(0.8, 1.2, N_bg)
    z_s  = np.concatenate([z_fg, z_bg])
    e1   = rng.uniform(-0.2, 0.2, N)
    e2   = rng.uniform(-0.2, 0.2, N)
    W    = np.ones(N)
    M    = np.zeros(N)
    sources = (ra, dec, z_s, e1, e2, W, M)
    result = d.lensfit_cluster_lensing(cluster, sources, radius=10.0)
    # no foreground galaxy should pass z_s > 1.1*0.5 + 0.1 = 0.65
    assert result['Count'] == N_bg
    assert np.all(result['Critical Density'] > 0)


# ---------------------------------------------------------------------------
# metacal_cluster_lensing
# ---------------------------------------------------------------------------

@pytest.fixture
def metacal_result():
    cluster = (0.0, 0.0, 0.3)
    sources = _make_sources_metacal(500, z_cluster=0.3, rng=np.random.default_rng(11))
    return d.metacal_cluster_lensing(cluster, sources, radius=10.0)


def test_metacal_returns_expected_keys(metacal_result):
    for key in ['Critical Density', 'Tangential Shear', 'Cross Shear',
                'Radial Distance', 'Polar Angle', 'Weights', 'Mult. Bias', 'Count']:
        assert key in metacal_result


def test_metacal_critical_density_positive(metacal_result):
    assert np.all(metacal_result['Critical Density'] > 0)


def test_metacal_count_consistent(metacal_result):
    N = metacal_result['Count']
    assert len(metacal_result['Critical Density']) == N
    assert len(metacal_result['Tangential Shear']) == N


def test_metacal_radial_distance_within_radius(metacal_result):
    assert np.all(metacal_result['Radial Distance'] <= 10.0)


def test_metacal_radial_distance_positive(metacal_result):
    assert np.all(metacal_result['Radial Distance'] > 0)


def test_metacal_foreground_excluded():
    # Mix foreground + background; foreground must not be counted.
    cluster = (0.0, 0.0, 0.5)
    rng = np.random.default_rng(21)
    N_fg, N_bg = 100, 50
    N = N_fg + N_bg
    ra  = rng.uniform(-0.003, 0.003, N)
    dec = rng.uniform(0.001, 0.003, N)
    z_fg = rng.uniform(0.1, 0.4, N_fg)
    z_bg = rng.uniform(0.9, 1.3, N_bg)
    z_s  = np.concatenate([z_fg, z_bg])
    e1   = rng.uniform(-0.2, 0.2, N)
    e2   = rng.uniform(-0.2, 0.2, N)
    W    = np.ones(N)
    R11, R12, R21, R22 = (np.full(N, 0.7), np.zeros(N), np.zeros(N), np.full(N, 0.7))
    sources = (ra, dec, z_s, e1, e2, W, R11, R12, R21, R22)
    result = d.metacal_cluster_lensing(cluster, sources, radius=10.0)
    # cut: z_s > 1.1*0.5 + 0.2 = 0.75; all z_bg > 0.9 pass, all z_fg < 0.4 fail
    assert result['Count'] == N_bg
    assert np.all(result['Critical Density'] > 0)


# ---------------------------------------------------------------------------
# signal
# ---------------------------------------------------------------------------

@pytest.fixture
def bin_limits():
    edges = np.linspace(0.1, 5.0, 7)
    return np.column_stack([edges[:-1], edges[1:]])  # (6, 2)


def test_signal_output_shape(bin_limits):
    stake = _make_stake(N=300)
    ds, dx = d.signal(stake, bin_limits)
    assert ds.shape == (len(bin_limits),)
    assert dx.shape == (len(bin_limits),)


def test_signal_zero_shear_zero_signal(bin_limits):
    stake = _make_stake(N=300)
    stake[1] = 0.0  # e_t = 0
    stake[2] = 0.0  # e_x = 0
    ds, dx = d.signal(stake, bin_limits)
    np.testing.assert_allclose(ds, 0.0, atol=1e-12)
    np.testing.assert_allclose(dx, 0.0, atol=1e-12)


def test_signal_matches_reference_formula(bin_limits):
    """ΔΣ in bin b = Σ(e_t·W/Σ) / Σ((1+M)·W/Σ²)"""
    stake = _make_stake(N=300)
    ds, _ = d.signal(stake, bin_limits)

    sig, et, w, R, M = stake[0], stake[1], stake[3], stake[4], stake[5]
    w_eff = w / sig**2

    for b, (r_lo, r_hi) in enumerate(bin_limits):
        mask = (R > r_lo) & (R < r_hi)
        if mask.sum() == 0:
            continue
        expected = (et[mask] * w[mask] / sig[mask]).sum() / ((1 + M[mask]) * w_eff[mask]).sum()
        np.testing.assert_allclose(ds[b], expected, rtol=1e-12)


def test_signal_empty_bin_gives_nan():
    # Single galaxy placed so only bin 0 is occupied
    sig = np.array([2.0])
    et  = np.array([0.1])
    ex  = np.array([0.05])
    W   = np.array([1.0])
    R   = np.array([0.15])
    M   = np.array([0.0])
    stake = np.vstack([sig, et, ex, W, R, M])
    bins = np.array([[0.1, 0.2], [1.0, 2.0]])  # galaxy only in bin 0
    ds, dx = d.signal(stake, bins)
    assert np.isfinite(ds[0])
    assert np.isnan(ds[1])


# ---------------------------------------------------------------------------
# stacked_signal
# ---------------------------------------------------------------------------

def _make_cluster_list(N_clusters=10, N_gal=150, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return [_make_stake(N=N_gal, rng=rng) for _ in range(N_clusters)]


def test_stacked_signal_output_shapes(bin_limits, capsys):
    clusters = _make_cluster_list()
    sigmas, boosts, cov_t, xigmas, cov_x = d.stacked_signal(clusters, bin_limits, Nboot=100)
    Nb = len(bin_limits)
    assert sigmas.shape == (Nb,)
    assert boosts.shape == (Nb,)
    assert cov_t.shape == (Nb, Nb)
    assert cov_x.shape == (Nb, Nb)


def test_stacked_signal_covariance_symmetric(bin_limits, capsys):
    clusters = _make_cluster_list()
    _, _, cov_t, _, cov_x = d.stacked_signal(clusters, bin_limits, Nboot=200)
    np.testing.assert_allclose(cov_t, cov_t.T, atol=1e-15)
    np.testing.assert_allclose(cov_x, cov_x.T, atol=1e-15)


def test_stacked_signal_zero_shear_zero_mean(bin_limits, capsys):
    rng = np.random.default_rng(99)
    clusters = []
    for _ in range(15):
        s = _make_stake(N=200, rng=rng)
        s[1] = 0.0  # e_t = 0
        s[2] = 0.0  # e_x = 0
        clusters.append(s)
    sigmas, _, _, xigmas, _ = d.stacked_signal(clusters, bin_limits, Nboot=100)
    np.testing.assert_allclose(sigmas, 0.0, atol=1e-10)
    np.testing.assert_allclose(xigmas, 0.0, atol=1e-10)


def test_stacked_signal_boosts_positive(bin_limits, capsys):
    clusters = _make_cluster_list()
    _, boosts, _, _, _ = d.stacked_signal(clusters, bin_limits, Nboot=50)
    assert np.all(boosts > 0)


# ---------------------------------------------------------------------------
# single_cluster
# ---------------------------------------------------------------------------

def test_single_cluster_output_shapes(bin_limits, capsys):
    clusters = [_make_stake(N=400)]
    sigmas, cov_t, xigmas, cov_x = d.single_cluster(clusters, bin_limits, Nboot=100)
    Nb = len(bin_limits)
    assert sigmas.shape == (Nb,)
    assert cov_t.shape == (Nb, Nb)
    assert cov_x.shape == (Nb, Nb)


def test_single_cluster_covariance_symmetric(bin_limits, capsys):
    clusters = [_make_stake(N=400)]
    _, cov_t, _, cov_x = d.single_cluster(clusters, bin_limits, Nboot=200)
    np.testing.assert_allclose(cov_t, cov_t.T, atol=1e-15)
    np.testing.assert_allclose(cov_x, cov_x.T, atol=1e-15)


def test_single_cluster_zero_shear_zero_mean(bin_limits, capsys):
    rng = np.random.default_rng(77)
    s = _make_stake(N=400, rng=rng)
    s[1] = 0.0
    s[2] = 0.0
    sigmas, _, xigmas, _ = d.single_cluster([s], bin_limits, Nboot=100)
    np.testing.assert_allclose(sigmas, 0.0, atol=1e-10)
    np.testing.assert_allclose(xigmas, 0.0, atol=1e-10)


def test_single_cluster_empty_bin_handled(capsys):
    # Put all galaxies in R=[0.1,0.2]; bins=[0.1,0.3] and [2.0,3.0] — second is empty
    rng = np.random.default_rng(55)
    N = 200
    sig = rng.uniform(1, 4, N)
    et  = rng.uniform(-0.2, 0.2, N)
    ex  = rng.uniform(-0.2, 0.2, N)
    W   = np.ones(N)
    R   = rng.uniform(0.1, 0.3, N)   # all in first bin
    M   = np.zeros(N)
    stake = np.vstack([sig, et, ex, W, R, M])
    bins = np.array([[0.1, 0.4], [2.0, 3.0]])
    sigmas, cov_t, _, _ = d.single_cluster([stake], bins, Nboot=50)
    # second bin should be nan; first should be finite
    assert np.isfinite(sigmas[0])
    assert np.isnan(sigmas[1])
