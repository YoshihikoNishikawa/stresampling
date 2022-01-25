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
from multiprocessing import Pool
import psutil


def build_pseudo_timeseries(prob, trajectory, iteration):
    """
    Resampling method for the stationary bootstrap

    Args:
        prob: The parameter p in (0,1) for the stationary bootstrap method
        trajectory: Input time series data
        iteration: The (minimum) number of sampling from a geometric distribution needed to fill the pseudo time series

    Returns:
        A single resampled time series
    """

    list_length = np.random.geometric(prob, size=iteration)
    while np.sum(list_length) < trajectory.shape[0]:
        list_length = np.append(
            list_length, np.random.geometric(prob, size=1))

    list_beginning = np.random.randint(
        low=0, high=trajectory.shape[0], size=list_length.shape[0]
    )
    # Find the maximum index to be accessed (which can be larger than the trajectory length)
    max_index = np.max(list_beginning + list_length)

    # Find the minimum number to repeat the trajectory
    number_repeat = int(np.ceil(max_index / trajectory.shape[0]))
    # Make a periodic trajectory by replicating the original one
    periodic_trajectory = trajectory
    for i in range(number_repeat):
        periodic_trajectory = np.concatenate((periodic_trajectory, trajectory))

    pseudo_time_series = np.zeros(np.sum(list_length))
    if trajectory.ndim > 1:
        pseudo_time_series = np.zeros(
            (np.sum(list_length), *trajectory[0].shape))
    first_index = 0
    for i in range(list_beginning.shape[0]):
        pseudo_time_series[first_index: first_index + list_length[i]] = periodic_trajectory[list_beginning[i]: list_beginning[i] +
                                                                                            list_length[i]]
        first_index += list_length[i]
    return pseudo_time_series[0:trajectory.shape[0]]


def output_bootstrap_samples(prob, trajectory, phys, number_bsamples):
    """
    single thread stationary bootstrap.
    The arguments are same with the single core version

    Args:
        prob: The parameter p for the stationary bootstrap method
        trajectory: The original time series data
        number_bsamples: Number of bootstrap samples

    Returns:
        resampled time series data
    """
    iteration = (int)(1.1 * trajectory.shape[0] * prob)

    bsamples_array = np.zeros(number_bsamples)
    for b_index in range(number_bsamples):
        bsamples_array[b_index] = phys(build_pseudo_timeseries(
            prob, trajectory, iteration)
        )
    return bsamples_array


def calc_phys(argument):
    phys = argument[0]
    prob = argument[1]
    trajectory = argument[2]
    iteration = argument[3]
    return phys(build_pseudo_timeseries(prob, trajectory, iteration))


def output_bootstrap_samples_parallel(prob, trajectory, phys, number_bsamples):
    """
    Parallelized stationary bootstrap.
    The arguments are same with the single core version

    Args:
        prob: The parameter p for the stationary bootstrap method
        trajectory: The original time series data
        number_bsamples: Number of bootstrap samples

    Returns:
        resampled time series data
    """

    iteration = int(1.1 * trajectory.shape[0] * prob)

    argument = (phys, prob, trajectory, iteration)
    with Pool(psutil.cpu_count(logical=False)) as p:
        bsamples_list = p.map(calc_phys, [argument]*number_bsamples)
    bsamples_array = np.array(bsamples_list)
    return bsamples_array
