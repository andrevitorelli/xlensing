"""Likelihood and prior factories for sampling Delta Sigma models.

Each factory returns a function of the parameter vector ``theta`` that can be
summed into a log-posterior and passed to a sampler (e.g. ``emcee``)::

    from functools import partial
    like = ln_gaussian_likelihood_maker((signal, cov), partial(my_model, radii=r))
    priors = [ln_flat_prior_maker(1e13, 1e16, 0), ln_gaussian_prior_maker(5., 2., 1)]
    def ln_post(theta):
        lp = sum(p(theta) for p in priors)
        return lp + like(theta) if np.isfinite(lp) else -np.inf
"""
import numpy as np


def ln_gaussian_likelihood_maker(data, model, hartlap=1):
    """Return a Gaussian log-likelihood ``ln L(theta)`` for a data vector and covariance.

    Parameters
    ----------
    data : tuple (signal, covariance)
        ``signal`` is an (N,) data vector and ``covariance`` its (N, N)
        covariance matrix.
    model : callable
        ``model(theta)`` returning the (N,) model prediction; bind any fixed
        arguments (radii, redshift, ...) beforehand, e.g. with
        ``functools.partial``.
    hartlap : float, optional
        Factor multiplying the inverse covariance, e.g. the Hartlap et al.
        (2007) debiasing factor (n - p - 2) / (n - 1) for a covariance
        estimated from n realisations.  Default 1.

    Returns
    -------
    callable
        ``ln_like(theta)``, the Gaussian log-likelihood including the
        normalisation; NaN results are returned as ``-inf``.
    """

    def ln_gaussian_like(theta):
        signal, covariance = data

        N = len(signal)

        model_eval = model(theta)  # model is m = f(theta, x) with x previously set by functools.partial

        inv_cov_estim = hartlap * np.linalg.inv(covariance)
        vec = signal - model_eval

        vec_row = vec.reshape(1, N)
        vec_col = vec.reshape(N, 1)

        lndetcov = np.linalg.slogdet(covariance)[1]

        result = -(np.dot(vec_row, np.dot(inv_cov_estim, vec_col))) - N * 1.837877 - lndetcov  # 1.837877 = ln(2 pi)

        if result != result:  # tests if result is nan
            result = -np.inf
        return result / 2
    return ln_gaussian_like


def ln_flat_prior_maker(low, high, pos):
    """Return a flat log-prior on parameter ``theta[pos]``: 0 inside (low, high), ``-inf`` outside.

    The prior is not normalised (its constant does not affect sampling).
    """
    def prior(theta):
        if low < theta[pos] < high:
            return 0
        else:
            return -np.inf
    return prior


def ln_gaussian_prior_maker(mu, sigma, pos):
    """Return a normalised Gaussian log-prior N(mu, sigma) on parameter ``theta[pos]``."""
    def prior(x):
        result = -(x[pos] - mu) * (x[pos] - mu) / (sigma ** 2)\
                 - 1.837877\
                 - 2 * np.log(sigma)
        return result / 2
    return prior
