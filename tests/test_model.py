import numpy as np
import pytest
from xlensing import model as m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def radii():
    return np.logspace(-1, 1, 8)  # 8 radii: 0.1 – 10 Mpc


@pytest.fixture
def single_cluster():
    return dict(
        M200m      = np.array([3e14]),
        C200m      = np.array([5.0]),
        Z          = np.array([0.3]),
        FMISS      = np.array([0.2]),
        SIGMA_OFF  = np.array([0.3]),
        BCG_B_MASS = np.array([5e11]),
    )


@pytest.fixture
def cluster_array():
    rng = np.random.default_rng(42)
    N = 20
    return dict(
        M200m      = rng.uniform(1e13, 1e15, N),
        C200m      = rng.uniform(3.0,  8.0,  N),
        Z          = rng.uniform(0.1,  0.8,  N),
        FMISS      = rng.uniform(0.0,  0.5,  N),
        SIGMA_OFF  = rng.uniform(0.1,  0.5,  N),
        BCG_B_MASS = rng.uniform(1e11, 1e12, N),
    )


# ---------------------------------------------------------------------------
# NFW_delta_c
# ---------------------------------------------------------------------------

def test_NFW_delta_c_positive():
    assert m.NFW_delta_c(5.0) > 0


def test_NFW_delta_c_increases_with_concentration():
    c = np.array([3.0, 5.0, 10.0])
    result = m.NFW_delta_c(c)
    assert np.all(np.diff(result) > 0)


def test_NFW_delta_c_vectorized():
    c = np.array([3.0, 5.0, 10.0])
    assert m.NFW_delta_c(c).shape == (3,)


# ---------------------------------------------------------------------------
# r_vir_c / r_vir_m
# ---------------------------------------------------------------------------

def test_r_vir_positive():
    assert m.r_vir_c(0.3, 1e14) > 0
    assert m.r_vir_m(0.3, 1e14) > 0


def test_r_vir_increases_with_mass():
    z = 0.3
    assert m.r_vir_m(z, 1e13) < m.r_vir_m(z, 1e14) < m.r_vir_m(z, 1e15)


def test_r_vir_m_larger_than_r_vir_c():
    # rhoM = OmegaM * rhoc < rhoc  →  r_vir_m > r_vir_c
    assert m.r_vir_m(0.3, 1e14) > m.r_vir_c(0.3, 1e14)


# ---------------------------------------------------------------------------
# SigX
# ---------------------------------------------------------------------------

def test_SigX_at_unity():
    np.testing.assert_allclose(m.SigX(np.array([1.0])), [1 / 3], rtol=1e-10)


def test_SigX_continuous_at_unity():
    eps = 1e-6
    vals = m.SigX(np.array([1 - eps, 1.0, 1 + eps]))
    np.testing.assert_allclose(vals[0], vals[1], rtol=1e-4)
    np.testing.assert_allclose(vals[2], vals[1], rtol=1e-4)


def test_SigX_positive():
    x = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    assert np.all(m.SigX(x) > 0)


def test_SigX_decreasing():
    x = np.array([0.5, 1.0, 2.0, 5.0])
    assert np.all(np.diff(m.SigX(x)) < 0)


# ---------------------------------------------------------------------------
# SigBar
# ---------------------------------------------------------------------------

def test_SigBar_at_unity():
    expected = 2.0 * (1 + np.log(0.5))
    np.testing.assert_allclose(m.SigBar(np.array([1.0])), [expected], rtol=1e-10)


def test_SigBar_continuous_at_unity():
    eps = 1e-6
    vals = m.SigBar(np.array([1 - eps, 1.0, 1 + eps]))
    np.testing.assert_allclose(vals[0], vals[1], rtol=1e-4)
    np.testing.assert_allclose(vals[2], vals[1], rtol=1e-4)


def test_SigBar_exceeds_SigX():
    # Mean within R > value at R for any monotonically decreasing profile.
    x = np.array([0.3, 0.5, 1.0, 2.0, 5.0])
    assert np.all(m.SigBar(x) > m.SigX(x))


# ---------------------------------------------------------------------------
# Growth factor D
# ---------------------------------------------------------------------------

def test_D_normalization():
    np.testing.assert_allclose(m.D(np.array([1.0])), [1.0], rtol=1e-6)


def test_D_monotonically_increasing():
    a = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.all(np.diff(m.D(a)) > 0)


def test_D_bounded():
    a = np.array([0.1, 0.5, 1.0])
    result = m.D(a)
    assert np.all(result > 0)
    assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# sigma_squared / sigma8
# ---------------------------------------------------------------------------

def test_sigma8_physical():
    # Allow generous tolerance since the test power spectrum may not be Planck18-exact.
    assert 0.7 < float(m.sigma8()) < 0.9


def test_sigma_squared_consistent_with_sigma8():
    np.testing.assert_allclose(m.sigma_squared(0.0, 8.0), m.sigma8() ** 2, rtol=1e-5)


def test_sigma_squared_decreasing_with_R():
    R = np.array([1.0, 4.0, 8.0, 20.0])
    result = np.array([float(m.sigma_squared(0.0, r)) for r in R])
    assert np.all(np.diff(result) < 0)


def test_sigma_squared_decreasing_with_z():
    # sigma² ∝ D(z)², which decreases with redshift.
    R = 8.0
    s = [float(m.sigma_squared(z, R)) for z in [0.0, 0.5, 1.0]]
    assert s[0] > s[1] > s[2]


# ---------------------------------------------------------------------------
# peak_nu / Tinker_bias / Bias
# ---------------------------------------------------------------------------

def test_peak_nu_positive():
    assert np.all(m.peak_nu(np.array([0.3]), np.array([1e14])) > 0)


def test_peak_nu_increases_with_mass():
    z = np.array([0.3, 0.3])
    nu = m.peak_nu(z, np.array([1e13, 1e15]))
    assert nu[0] < nu[1]


def test_peak_nu_increases_with_z():
    # Higher z → smaller D → smaller sigma → higher nu for fixed mass.
    M = np.array([1e14, 1e14])
    nu = m.peak_nu(np.array([0.1, 1.0]), M)
    assert nu[0] < nu[1]


def test_Tinker_bias_positive():
    assert np.all(m.Tinker_bias(np.array([0.5, 1.0, 2.0, 4.0])) > 0)


def test_Tinker_bias_increases_with_nu():
    nu = np.array([0.5, 1.0, 2.0, 4.0])
    assert np.all(np.diff(m.Tinker_bias(nu)) > 0)


def test_Bias_positive():
    assert np.all(m.Bias(np.array([0.3]), np.array([1e14])) > 0)


def test_Bias_increases_with_mass():
    z = np.array([0.3, 0.3])
    b = m.Bias(z, np.array([1e13, 1e15]))
    assert b[0] < b[1]


def test_Bias_shape():
    N = 10
    result = m.Bias(np.full(N, 0.3), np.full(N, 1e14))
    assert result.shape == (N,)


# ---------------------------------------------------------------------------
# Delta_Sigma_l
# ---------------------------------------------------------------------------

def test_Delta_Sigma_l_shape():
    N, M = 5, 8
    result = m.Delta_Sigma_l(np.full(N, 0.3), np.logspace(-1, 1, M))
    assert result.shape == (N, M)


def test_Delta_Sigma_l_positive():
    z = np.array([0.1, 0.3, 0.5])
    R = np.array([0.5, 1.0, 5.0])
    assert np.all(m.Delta_Sigma_l(z, R) > 0)


# ---------------------------------------------------------------------------
# Delta_Sigma_NFW_off_x
# ---------------------------------------------------------------------------

def test_Delta_Sigma_NFW_off_x_shape():
    N, M = 10, 8
    matrix = np.full((N, M), 0.5)
    vector = np.full(N, 0.3)
    result = m.Delta_Sigma_NFW_off_x(matrix, vector)
    assert result.shape == (N, M)


# ---------------------------------------------------------------------------
# NFW_Delta_Sigma — shapes
# ---------------------------------------------------------------------------

def test_NFW_Delta_Sigma_output_shapes(single_cluster, radii):
    N, M = 1, len(radii)
    result = m.NFW_Delta_Sigma(**single_cluster, radii=radii)
    for key in ['Signal', 'BCG Signal', 'NFW Signal', 'Miscentered Signal', 'Two-halo term']:
        assert result[key].shape == (N, M), f"'{key}' shape is {result[key].shape}"
    assert result['radii'] is radii


def test_NFW_Delta_Sigma_output_shapes_N_clusters(cluster_array, radii):
    N, M = len(cluster_array['M200m']), len(radii)
    result = m.NFW_Delta_Sigma(**cluster_array, radii=radii)
    for key in ['Signal', 'BCG Signal', 'NFW Signal', 'Miscentered Signal', 'Two-halo term']:
        assert result[key].shape == (N, M), f"'{key}' shape is {result[key].shape}"


# ---------------------------------------------------------------------------
# NFW_Delta_Sigma — physical content
# ---------------------------------------------------------------------------

def test_NFW_Delta_Sigma_components_positive(cluster_array, radii):
    result = m.NFW_Delta_Sigma(**cluster_array, radii=radii)
    for key in ['BCG Signal', 'NFW Signal', 'Miscentered Signal', 'Two-halo term']:
        assert np.all(result[key] > 0), f"'{key}' contains non-positive values"


def test_NFW_Delta_Sigma_total_signal_positive(cluster_array, radii):
    result = m.NFW_Delta_Sigma(**cluster_array, radii=radii)
    assert np.all(result['Signal'] > 0)


def test_NFW_Delta_Sigma_signal_is_weighted_sum_of_components(cluster_array, radii):
    result = m.NFW_Delta_Sigma(**cluster_array, radii=radii)
    FMISS = cluster_array['FMISS'][:, np.newaxis]
    expected = (result['BCG Signal']
                + (1 - FMISS) * result['NFW Signal']
                + FMISS       * result['Miscentered Signal']
                + result['Two-halo term'])
    np.testing.assert_allclose(result['Signal'], expected, rtol=1e-12)


def test_NFW_Delta_Sigma_fmiss_zero_drops_miscentered(single_cluster, radii):
    p = {**single_cluster, 'FMISS': np.array([0.0])}
    result = m.NFW_Delta_Sigma(**p, radii=radii)
    expected = result['BCG Signal'] + result['NFW Signal'] + result['Two-halo term']
    np.testing.assert_allclose(result['Signal'], expected, rtol=1e-12)


def test_NFW_Delta_Sigma_fmiss_one_drops_centered_nfw(single_cluster, radii):
    p = {**single_cluster, 'FMISS': np.array([1.0])}
    result = m.NFW_Delta_Sigma(**p, radii=radii)
    expected = result['BCG Signal'] + result['Miscentered Signal'] + result['Two-halo term']
    np.testing.assert_allclose(result['Signal'], expected, rtol=1e-12)


def test_NFW_Delta_Sigma_higher_mass_larger_signal(radii):
    common = dict(
        C200m      = np.array([5.0, 5.0]),
        Z          = np.array([0.3, 0.3]),
        FMISS      = np.array([0.0, 0.0]),
        SIGMA_OFF  = np.array([0.3, 0.3]),
        BCG_B_MASS = np.array([5e11, 5e11]),
    )
    result = m.NFW_Delta_Sigma(M200m=np.array([1e13, 1e15]), radii=radii, **common)
    assert np.all(result['Signal'][1] > result['Signal'][0])


# ---------------------------------------------------------------------------
# NFW_Delta_Sigma — vectorisation consistency
# ---------------------------------------------------------------------------

def test_NFW_Delta_Sigma_single_matches_array_row(cluster_array, radii):
    # Each row of the N-cluster result must equal the single-cluster evaluation.
    result_array = m.NFW_Delta_Sigma(**cluster_array, radii=radii)
    for i in range(5):  # check first 5 clusters
        single = {k: cluster_array[k][i:i+1] for k in cluster_array}
        result_single = m.NFW_Delta_Sigma(**single, radii=radii)
        np.testing.assert_allclose(
            result_single['Signal'][0],
            result_array['Signal'][i],
            rtol=1e-12,
            err_msg=f"Mismatch at cluster index {i}",
        )
