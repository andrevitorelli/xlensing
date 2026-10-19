"""Tests for fitting, sims.sample_triaxial_shapes, the legacy testing helpers,
the miscentring-table builder and cosmological helpers."""
import numpy as np
import pytest

import xlensing
from xlensing import cosmo, fitting, model, sims, testing, data
from xlensing import build_miscentering_table as bmt


# ---------------------------------------------------------------------------
# package
# ---------------------------------------------------------------------------

def test_version_string():
    assert isinstance(xlensing.__version__, str) and xlensing.__version__.count(".") == 2


# ---------------------------------------------------------------------------
# fitting
# ---------------------------------------------------------------------------

def test_gaussian_likelihood_matches_scipy():
    from scipy.stats import multivariate_normal
    rng = np.random.default_rng(1)
    A = rng.normal(size=(4, 4))
    cov = A @ A.T + 4 * np.eye(4)
    y = rng.normal(size=4)
    like = fitting.ln_gaussian_likelihood_maker((y, cov), lambda th: th[0] * np.ones(4))
    for t in (0.0, 0.7, -1.3):
        ref = multivariate_normal(mean=t * np.ones(4), cov=cov).logpdf(y)
        np.testing.assert_allclose(float(np.squeeze(like([t]))), ref, rtol=1e-5)


def test_gaussian_likelihood_hartlap_scales_chi2():
    cov = np.eye(3)
    y = np.array([1.0, 2.0, 3.0])
    l1 = fitting.ln_gaussian_likelihood_maker((y, cov), lambda th: np.zeros(3), hartlap=1.0)
    l2 = fitting.ln_gaussian_likelihood_maker((y, cov), lambda th: np.zeros(3), hartlap=0.5)
    chi2 = 14.0
    np.testing.assert_allclose(float(np.squeeze(l1([0]) - l2([0]))), -0.5 * chi2 * 0.5, rtol=1e-12)


def test_gaussian_likelihood_nan_model_gives_minus_inf():
    like = fitting.ln_gaussian_likelihood_maker((np.zeros(2), np.eye(2)), lambda th: np.array([np.nan, 0.0]))
    assert np.squeeze(like([0])) == -np.inf


def test_flat_prior():
    p = fitting.ln_flat_prior_maker(0.0, 1.0, 1)
    assert p([5.0, 0.5]) == 0
    assert p([5.0, 1.5]) == -np.inf
    assert p([5.0, 0.0]) == -np.inf


def test_gaussian_prior_matches_scipy():
    from scipy.stats import norm
    p = fitting.ln_gaussian_prior_maker(2.0, 0.5, 0)
    for x in (1.0, 2.0, 3.3):
        np.testing.assert_allclose(p([x]), norm(2.0, 0.5).logpdf(x), rtol=1e-6)


# ---------------------------------------------------------------------------
# cosmology helpers and the two-halo term
# ---------------------------------------------------------------------------

def test_DA_matches_astropy():
    np.testing.assert_allclose(cosmo.DA(0, 0.5), cosmo.cosmology.angular_diameter_distance(0.5).value)


def test_rhocz_today_matches_astropy():
    rho = cosmo.cosmology.critical_density0.to("Msun/Mpc3").value
    np.testing.assert_allclose(cosmo.rhocz(0.0), rho, rtol=2e-3)


def test_growth_factor_normalised_and_decreasing():
    np.testing.assert_allclose(model.D(1.0), 1.0, rtol=1e-6)
    assert model.D(0.5) < model.D(0.8) < model.D(1.0)


def test_bias_increases_with_mass():
    z = np.array([0.3, 0.3])
    b = model.Bias(z, np.array([1e14, 1e15]))
    assert b[1] > b[0] > 0


def test_two_halo_term_positive_and_shape():
    z = np.array([0.2, 0.5])
    r = np.logspace(-0.5, 1, 7)
    dsl = model.Delta_Sigma_l(z, r)
    assert dsl.shape == (2, 7)
    assert np.all(dsl > 0)


def test_nfw_total_is_sum_of_terms():
    r = np.logspace(-1, 1, 9)
    out = model.NFW_Delta_Sigma(np.array([2e14]), np.array([4.0]), np.array([0.4]),
                                np.array([0.3]), np.array([0.4]), np.array([1e12]), r)
    tot = (out["BCG Signal"] + 0.7 * out["NFW Signal"] + 0.3 * out["Miscentered Signal"]
           + out["Two-halo term"])
    np.testing.assert_allclose(out["Signal"], tot, rtol=1e-12)


def test_sigmacrit_decreases_with_source_redshift():
    s = data.sigmacrit(0.3, np.array([0.6, 1.0, 1.5]))
    assert np.all(np.diff(s) < 0)


def test_tangential_response_identity():
    for phi in (0.0, 0.3, 1.1):
        np.testing.assert_allclose(data.tangential_response(1.0, 0.0, 0.0, 1.0, phi), 1.0)


# ---------------------------------------------------------------------------
# miscentring table builder
# ---------------------------------------------------------------------------

def test_miscentering_builder_matches_shipped_table():
    x = np.array([0.3, 1.0, 3.0])
    xi = 0.5
    built = bmt.delta_sigma_mis(x, xi, n_fine=3000)
    table = model.Delta_Sigma_NFW_off_x(x[None, :], np.array([xi]))[0]
    np.testing.assert_allclose(built, table, rtol=2e-2, atol=1e-4)


def test_miscentering_small_offset_tends_to_centred():
    x = np.array([0.5, 1.0, 2.0])
    centred = model.SigBar(x) - model.SigX(x)
    np.testing.assert_allclose(bmt.delta_sigma_mis(x, 1e-3, n_fine=3000), centred, rtol=1e-2)


# ---------------------------------------------------------------------------
# sims.sample_triaxial_shapes
# ---------------------------------------------------------------------------

def test_sample_triaxial_shapes_ranges():
    rng = np.random.default_rng(3)
    z = np.full(50, 0.3)
    M = np.full(50, 1e14)
    out = sims.sample_triaxial_shapes(z, M, rng=rng)
    assert np.all((out["q_ba"] >= 0.5) & (out["q_ba"] <= 1.0))
    assert np.all(out["q_ca"] <= out["q_ba"])
    assert np.all(out["q_min"] <= out["q_maj"] + 1e-12)
    assert np.all(out["q_maj"] <= out["r_vir"] + 1e-9)


def test_sample_triaxial_shapes_length_mismatch():
    with pytest.raises(ValueError):
        sims.sample_triaxial_shapes(np.array([0.3, 0.4]), np.array([1e14]))


def test_sample_triaxial_shapes_fixed_los():
    out = sims.sample_triaxial_shapes(np.array([0.3]), np.array([1e14]), isotropic_los=False,
                                      rng=np.random.default_rng(0))
    assert out["theta_los"][0] == 0.0


# ---------------------------------------------------------------------------
# legacy testing helpers
# ---------------------------------------------------------------------------

def test_truncnorm_bounds():
    np.random.seed(0)
    x = testing.truncnorm(-0.2, 0.2, 0.0, 0.3, 500)
    assert x.shape == (500,) and np.all(np.abs(x) < 0.2)


def test_gen_gal_shapes():
    np.random.seed(1)
    e1, e2, ra, dec, z = testing.gen_gal(Ngals=300)
    assert all(len(a) == 300 for a in (e1, e2, ra, dec, z))


def test_gnfw_continuous_at_one():
    np.testing.assert_allclose(testing.gNFW(1.0 - 1e-6), testing.gNFW(1.0), rtol=1e-4)
    np.testing.assert_allclose(testing.gNFW(1.0 + 1e-6), testing.gNFW(1.0), rtol=1e-4)


def test_legacy_nfw_shear_only_shears_background():
    np.random.seed(2)
    e1, e2, ra, dec, z = testing.gen_gal(Ngals=200, Zcluster=0.3)
    cluster = (3e14, 5.0, 0.0, 0.0, 0.3)
    eps = testing.NFW_shear(cluster, (ra, dec, z, e1, e2))
    fg = z <= 0.3
    np.testing.assert_allclose(eps[fg], (e1 + 1j * e2)[fg])
    assert np.any(np.abs(eps[~fg] - (e1 + 1j * e2)[~fg]) > 0)
