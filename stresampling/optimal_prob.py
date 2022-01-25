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
from scipy import signal


def calculate_autocorrelation(trajectory):
    """ 
    Calculate the autocorrelation function
    Arg:
        trajectory: Input time series data
    Return:
        The autocorrelation function
    """
    mean_data = np.mean(trajectory)
    if trajectory.ndim > 1:
        mean_data = np.full(trajectory.shape, np.mean(trajectory, axis=0))

    if trajectory.ndim > 1:
        autocorrelation = np.array(np.sum(
            np.array([signal.correlate(
                trajectory[(slice(None), *i)] - mean_data[(slice(None), *i)],
                trajectory[(slice(None), *i)] - mean_data[(slice(None), *i)],
                mode='full', method='auto') / trajectory.shape[0]
                for i, value in np.ndenumerate(trajectory[0])]), axis=0)
        )
        autocorrelation = autocorrelation[trajectory.shape[0] - 1:]
    else:
        autocorrelation = signal.correlate(
            trajectory - mean_data, trajectory - mean_data, mode='full', method='auto') / trajectory.shape[0]
        autocorrelation = autocorrelation[trajectory.shape[0] - 1:]

    return autocorrelation


def find_bandwidth(autocorrelation):
    """
    Find the bandwidth from the autocorrelation function, a la [D. N. Politis (2003)]
    Arg:
        autocorrelation: The autocorrelation function of a timeseries
    Return:
        two times the estimated bandwidth
    """
    c = 2
    threshold = c * \
        np.sqrt(np.log10(len(autocorrelation)) / len(autocorrelation))
    K = np.max([5, int(np.sqrt(np.log10(len(autocorrelation))))])

    normalized_autocorrelation = autocorrelation / autocorrelation[0]
    list_index = np.array(
        np.where(abs(normalized_autocorrelation) < threshold))
    list_index = list_index.reshape(list_index.shape[1])

    list_time = np.array([j for j in list_index for i in range(K)
                          if j + i < normalized_autocorrelation.shape[0] and abs(
        normalized_autocorrelation[j + i]) < threshold])

    bandwidth = np.min(list_time)

    return 2 * bandwidth


def calculate_p_opt(trajectory):
    """
    Calculate an optimal probability from a timeseries
    Arg:
        trajectory: Input time series data
    Returns:
        p_opt: Estimated optimal probability
        autocorrelation: The autocorrelation function of a timeseries
    """
    autocorrelation = calculate_autocorrelation(trajectory)
    bandwidth = find_bandwidth(autocorrelation)

    list_G = np.array([window_function(lag, bandwidth) * lag *
                       autocorrelation[lag] for lag in range(bandwidth)])
    G = 2 * np.sum(list_G)

    list_D = np.array([window_function(lag, bandwidth) *
                       autocorrelation[lag] for lag in range(1, bandwidth)])
    D = 2 * (autocorrelation[0] + 2 * np.sum(list_D)) ** 2.0

    p_opt = np.power((2.0 * G ** 2.0 / D) *
                     autocorrelation.shape[0], -1.0 / 3.0)

    if p_opt > 1.0:
        return 1.0, autocorrelation
    else:
        return p_opt, autocorrelation


def window_function(lag, bandwidth):
    """
    A window function used to integrate the autocorrelation function
    """
    if lag <= 0.5 * bandwidth:
        return 1.0
    elif lag < bandwidth:
        return 2.0 * (1.0 - lag / bandwidth)
    else:
        return 0
