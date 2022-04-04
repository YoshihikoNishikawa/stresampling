#
#   Copyright (c) 2022, Yoshihiko Nishikawa, Jun Takahashi, and Takashi Takahashi
#   Date: Apr. 2022
#
#   Python3 package for statistical analysis of stationary timeseries using the stationary bootstrap method [D. N. Politis and J. P. Romano (1994)]
#   with estimating an optimal parameter [D. N. Politis and H. White (2004)][A. Patton, D. N. Politis, and H. White (2009)].
#
#   URL: https://github.com/YoshihikoNishikawa/stresampling
#   See LICENSE for copyright information
#
#   If you use this code or find it useful, please cite arXiv:2112.11837
#

import numpy as np
from scipy.stats import norm


def percentile(list_observable, alpha):
    """
    The bootstrap percentile method for estimating 100*alpha % confidence interval

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence

    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """

    gap = 0.5 * (1.0 - alpha)
    return np.quantile(list_observable, gap), np.quantile(list_observable, 1.0 - gap) 


def bias_corrected(list_observable, alpha, mean):
    """
    The bias-corrected (BC) method for estimating the confidence interval,
    see [T. J. Diccio and B. Efron (1996)]

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence
        mean: The empirical mean value
    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """
    hatalpha0 = np.count_nonzero(
        list_observable <= mean) / len(list_observable)
    hatz0 = norm.ppf(hatalpha0)

    gap = 0.5 * (1.0 - alpha)
    acceleration = 0
    l_hatalpha = norm.cdf(hatz0 + (hatz0 + norm.ppf(gap)) /
                          (1.0 - acceleration * (hatz0 + norm.ppf(gap))))
    u_hatalpha = norm.cdf(hatz0 + (hatz0 + norm.ppf(1.0 - gap)) /
                          (1.0 - acceleration * (hatz0 + norm.ppf(1.0 - gap))))
    return np.quantile(list_observable, l_hatalpha), np.quantile(list_observable, u_hatalpha) 


def Bootstrap_t(list_observable, alpha, mean):
    """
    The bootstrap-t method for estimating the confidence interval, with an assumption the standard error
    does not depend on bootstrap samples, see [Gotze and Kunsch (1991)][J. P. Romano and M. Wolf (2006)],
    and [T. J. Diccio and B. Efron (1996)] for the iid case.

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence
        mean: The empirical mean value
    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """
    gap = 0.5 * (1.0 - alpha)
    return 2.0 * mean - np.quantile(list_observable, 1.0 - gap), 2.0 * mean - np.quantile(list_observable, gap)


def sym_Bootstrap_t(list_observable, alpha, mean):
    """
    The symmetric bootstrap-t method for estimating the confidence interval, with an assumption the standard error
    does not depend on bootstrap samples, see [P. Hall (1988)] and [J. P. Romano and M. Wolf (2006)].

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence
        mean: The empirical mean value
    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """
    list_t = np.abs(list_observable - mean)
    quantile = np.quantile(list_t, alpha)
    return mean - quantile, mean + quantile
