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
from .optimal_prob import *
from .build_bootstrap_samples import *
from .confidence_interval import *


class Stat:
    """A class for output of the stationary bootstrap analysis

    Attributes:
        mean: The estimate
        se: The standard error of the estimate
        low and up: The lower and upper confidence limits
        prob: The estimated optimal probability for the stationary bootstrap method 
        dist: array of the bootstrap samples of the quantity
        autocorr: array for the autocorrelation function of the timeseries
    """

    def __init__(self):
        self.mean = 0
        self.se = 0
        self.low = 0
        self.up = 0
        self.prob = 0
        self.dist = np.zeros(0)
        self.autocorr = np.zeros(0)


def conf_int(trajectory, phys, alpha, number_bsamples=1000, *, parallel=True, method='percentile'):
    """stationary bootstrap method

    Args:
        trajectory: input time series of shape (series_length, dimension)
        phys: Definition of the physical value of interest
        this method takes a time-series data as an argument and returns the physical value of interest
        number_bsamples: the number of bootstrap samples
        alpha: confidence level in [0.0, 1.0]
        parallel: parallelization flag
        method: {'percentile', 'bt', 'symbt', 'bca'}, default='percentile'.
        Used to specify the method to compute the confidence interval.

    Returns:
        Estimation of the physcal quantity, its standard error,
        the lower and upper limits of the confidence interval (estimated using the chosen method),
        and the optimal value of the probability used in the method
    """

    if not isinstance(trajectory, np.ndarray):
        trajectory = np.array(trajectory)

    if trajectory.ndim > 1:
        if trajectory.shape[1] > trajectory.shape[0]:
            print('Warning: The input time-series length is smaller than the other dimensions. The time direction of the time series must be in the x axis.')

    output = Stat()

    output.prob, output.autocorr = calculate_p_opt(trajectory)
    if parallel:
        list_phys = output_bootstrap_samples_parallel(
            output.prob, trajectory, phys, number_bsamples)
    else:
        list_phys = output_bootstrap_samples(
            output.prob, trajectory, phys, number_bsamples)

    simple_mean = phys(trajectory)
    output.mean = simple_mean
    output.se = np.sqrt(
        number_bsamples * np.var(list_phys) / (number_bsamples - 1))
    output.dist = list_phys

    if method == 'percentile':
        output.low, output.up = percentile(
            list_phys, alpha)
    elif method == 'bt':
        output.low, output.up = Bootstrap_t(
            list_phys, alpha, simple_mean)
    elif method == 'symbt':
        output.low, output.up = sym_Bootstrap_t(
            list_phys, alpha, simple_mean)
    elif method == 'bc':
        output.low, output.up = bias_corrected(
            list_phys, alpha, simple_mean)
    else:
        print('Error, choose one of percentile, bt (bootstrap-t), symbt (symmetric bootstrap-t), or bc (bias-corrected) methods.')

    return output
