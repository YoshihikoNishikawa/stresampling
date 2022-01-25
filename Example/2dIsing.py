#
#   Copyright (c) 2022, Yoshihiko Nishikawa, Jun Takahashi, and Takashi Takahashi
#   Date: * Jan. 2022
#
#   Python3 code for Monte Carlo simulation of the critical two-dimensional Ising model with the Metropolis algorithm.
#   The statistics of the second-moment correlation length is calculated with the stationary bootstrap method [D. N. Politis and J. P. Romano (1994)]
#   using the stresampling package.
#
#   URL: https://github.com/YoshihikoNishikawa/stresampling
#   See LICENSE for copyright information
#
#   If you use this code or find it useful, please cite arXiv:2112.11837
#

import numpy as np
from stresampling import stationary_bootstrap as sbm
import scipy.fft


def Metropolis(spins):
    """The Metopolis algorithm for the two-dimensional Ising model at the critical temperature

    Args:
        spins: The current spin configuration of the system
    Returns:
        A spin configuration after one Monte Carlo sweep

    """
    # Critical inverse temperature
    beta_c = 1.0 / (2.0 / (np.log(1.0 + np.sqrt(2.0))))

    for i in range(spins.shape[0]):
        for j in range(spins.shape[1]):
            de = 2 * spins[i, j] * (spins[(i + 1) % spins.shape[0], j] + spins[i, (j + 1) % spins.shape[1]] +
                                    spins[(i - 1 + spins.shape[0]) % spins.shape[0], j] +
                                    spins[i, (j - 1 + spins.shape[1]) % spins.shape[1]])
            prob = 1.0
            if de > 0.0:
                prob = np.exp(-beta_c * de)
            if np.random.rand() < prob:
                spins[i, j] = -spins[i, j]

    return spins


def Set_init_conf(spins):
    """ Set an initial configuration for the system

    Args:
        spins: The spin configuration of the system
    Returns:
        A spin configuration at the high-temperature limit
    """
    return 2 * (np.random.randint(0, 2, spins.shape) - 0.5)


def calc_susceptibility(spins):
    """ Calculate the magnetic susceptibility, or structure factor, of a spin configuration using FFT, up to a constant factor

    Args:
        spins: The current spin configuration of the system

    Returns:
        The structure factor of the spin configuration up to a constant
    """
    trans = scipy.fft.fft2(spins)
    trans = np.real(np.multiply(trans, np.conj(trans)))
    return trans


def calc_corr_length(seq):
    """ Calculate the 2nd moment correlation length

    Args:
        seq: A timeseries of the magnetic susceptibility

    Returns:
        The 2nd moment correlation length up to a constant factor

    """
    suscep = np.mean(seq, axis=0)
    L = suscep.shape[0]
    return np.sqrt(suscep[0, 0] / suscep[0, 1] - 1.0) / 2.0 / np.sin(np.pi / L) / L


def main():
    # Set the system size and the number of Monte Carlo sweeps per spin
    L = 8
    MaxMCS = 10000
    print('The critical temperature', (2.0 / (np.log(1.0 + np.sqrt(2.0)))))

    # Set an initial spin configuration
    spins = np.zeros((L, L))
    spins = Set_init_conf(spins)

    # Initial equilibration of the system
    for MCS in range(10000):
        Metropolis(spins)

    # Generate a timeseries of the structure factor
    seq = np.array([calc_susceptibility(Metropolis(spins))
                   for MCS in range(MaxMCS)])

    # Set parameters for the conf_int function
    alpha = 0.68
    stat = sbm.conf_int(seq, calc_corr_length, alpha,
                        parallel=True, method='bt')

    # Output the statistics
    print('Mean:', stat.mean, '\nStandard error:', stat.se, '\nLower limit:',
          stat.low, '\nUpper limit:', stat.up, '\nOptimal probability:', stat.prob)


if __name__ == "__main__":
    main()
