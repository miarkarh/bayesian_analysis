"""
The log-probability formulas.

@author: Mikko Artturi Karhunen
"""

import numpy as np


def log_prior_flat(theta, par_limits=None):
    """
    Check if parameter vector is inside the limits and return -inf likelihood if not.

    Outside of the limits is 0 probability, meaning -inf log probability.

    """
    # If no limits, no flat prior.
    if par_limits is None: return 0
    par_limits = np.array(par_limits)
    for i in range(np.size(theta)):
        if theta[i] < par_limits[i, 0] or par_limits[i, 1] < theta[i]:
            return -np.inf  # log(0)
    return 0


def log_likelihood(y, y_model, theta, cov_y=None, ystd=None, model_error=True):
    """
    Actually calculates only -chi^2, because normalization doesn't affect the MCMC sampling.

    Returns model's -chi^2 value based on the parameter vector, the experimental data, the model
    and their errors or covariances. This does not have normalization.

    Parameters
    ----------
    y : array
        Experimental values at datapoints.
    y_model : funcion
        The model which calculates predictions at each datapoint.
    theta : array
        The parameter vector passed to the model.
    cov_y : array, optional
        The experimental covariance matrix. The default is None, but then ystd must be given.
    ystd : array, optional
        The standard deviations of the experimental values at each datapoint. The default is None,
        but then cov_y must be given.
    model_error : bool, optional
        Defines if the model returns error or covariance matrix. The default is True.

    Raises
    ------
    Exception
        If neither of experimental error or covariance was given.

    Returns
    -------
    float
        Actually just the chi^2. Log-likelihood without normalization.
    """
    if model_error:
        y_m, err_m = y_model(theta)
        if err_m.ndim == 1:
            cov_m = np.diag(err_m**2)
        else:
            cov_m = err_m
    else:
        y_m = y_model(theta)
        cov_m = 0
    # If covariance data was given with experimental data.
    if cov_y is not None:
        # Covariance from the model is not yet implemented.
        cov = cov_y + cov_m
    elif ystd is not None:
        cov = np.diag(ystd**2) + cov_m  # Combines experimental and model errors.
    else:
        raise Exception("No experimental error given.")

    invcov = np.linalg.inv(cov)
    # L, info = np.linalg.lapack.dpotrf(cov, clean=False)
    # diff = y - y_m
    # alpha, info = np.linalg.lapack.dpotrs(cov, diff)
    # return -.5*np.dot(diff, alpha) - np.log(L.diagonal()).sum() # This would be -0.5 ( chi2 + log(det(cov)) )

    chi2 = (y - y_m).T @ invcov @ (y - y_m)
    return -chi2


def log_posterior(y, y_model, theta, par_limits=None, cov_y=None, ystd=None, model_error=True):
    """
    Combine flat prior and log-likelihood and calculates likelihood posterior.

    This does not have normalization implemented, because this is only used for MCMC.

    Parameters
    ----------
    y : array
        Experimental values at datapoints.
    y_model : funcion
        The model which calculates predictions at each datapoint.
    theta : array
        The parameter vector passed to the model.
    par_limits : array, optional
        The min max limits for each parameter in the parameter vector. The default is None.
    cov_y : array, optional
        The experimental covariance matrix. The default is None, but then ystd must be given.
    ystd : array, optional
        The standard deviations of the experimental values at each datapoint. The default is None,
        but then cov_y must be given.
    model_error : bool, optional
        Defines if the model returns error or covariance matrix. The default is True.

    Returns
    -------
    float
        Actually just the chi^2. Log-posterior without normalization.
    """
    lp = log_prior_flat(theta, par_limits)
    # This is for avoiding unnecessary computation of log-likelihood when flat prior is -infinite.
    if np.isinf(lp):
        return -np.inf
    ll = log_likelihood(y, y_model, theta, cov_y, ystd, model_error)
    return ll + lp
