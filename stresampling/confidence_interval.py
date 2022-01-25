#
#   Copyright (c) 2022, Yoshihiko Nishikawa, Jun Takahashi, and Takashi Takahashi
#   Date: * Jan. 2022
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


def percentile_conf_interval(list_observable, alpha, number_bsamples):
    """
    The bootstrap percentile method for estimating 100*alpha % confidence interval

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence

    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """

    list_observable.sort()
    gap = 0.5 * (1.0 - alpha)
    low_index = int(number_bsamples * gap)
    up_index = int(number_bsamples * (1.0 - gap))
    return list_observable[low_index], list_observable[up_index]


def lazy_BCa(list_observable, alpha, number_bsamples):
    """
    The bias-corrected and accelerated (BC_a) method with no acceleration for estimating the confidence interval,
    see [T. J. Diccio and B. Efron (1996)]

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence

    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """
    list_observable.sort()
    mean = np.mean(list_observable)
    hatalpha0 = np.count_nonzero(
        list_observable <= mean) / len(list_observable)
    hatz0 = norm.ppf(hatalpha0)

    gap = 0.5 * (1.0 - alpha)
    acceleration = 0
    l_hatalpha = norm.cdf(hatz0 + (hatz0 + norm.ppf(gap)) /
                          (1.0 - acceleration * (hatz0 + norm.ppf(gap))))
    u_hatalpha = norm.cdf(hatz0 + (hatz0 + norm.ppf(1.0 - gap)) /
                          (1.0 - acceleration * (hatz0 + norm.ppf(1.0 - gap))))
    low_index = int(number_bsamples * l_hatalpha)
    up_index = int(number_bsamples * u_hatalpha)
    return list_observable[low_index], list_observable[up_index]


def lazy_Bootstrap_t(list_observable, alpha, number_bsamples):
    """
    The bootstrap-t method for estimating the confidence interval, with an assumption the standard error
    does not depend on bootstrap samples, see [Gotze and Kunsch (1991)][J. P. Romano and M. Wolf (2006)],
    and [T. J. Diccio and B. Efron (1996)] for the iid case.

    Args:
        list_observable: A list of bootstrap samples
        alpha: The value for the confidence

    Return:
        The lower and upper limits of the estimated 100*alpha % confidence interval
    """
    list_observable.sort()
    sigma = np.sqrt(np.var(list_observable) / (number_bsamples - 1))
    mean = np.mean(list_observable)
    bmean = np.full(number_bsamples, mean)
    list_t = (list_observable - bmean) / sigma
    list_t.sort()
    gap = 0.5 * (1.0 - alpha)
    low_index = int(number_bsamples * gap)
    up_index = int(number_bsamples * (1.0 - gap))
    low_t = list_t[low_index]
    up_t = list_t[up_index]

    return mean + sigma * low_t, mean + sigma * up_t
