#
#   Copyright (c) 2022, Yoshihiko Nishikawa, Jun Takahashi, and Takashi Takahashi
#   Date: Apr. 2022
#
#   Python3 code for autoregressive models with a finite order.
#   The statistics of the covariance between two indep. timeseries is calculated with the stationary bootstrap method [D. N. Politis and J. P. Romano (1994)]
#   using the stresampling package.
#
#   URL: https://github.com/YoshihikoNishikawa/stresampling
#   See LICENSE for copyright information
#
#   If you use this code or find it useful, please cite arXiv:2112.11837
#


import numpy as np
from stresampling import stationary_bootstrap as sbm

np.random.seed(100)  # Fixed seed for PRNG


def covariance(seq):
    return np.dot(seq[:, 0], seq[:, 1]) / seq.shape[0] - np.mean(seq[:, 0]) * np.mean(seq[:, 1])


def lin_autoregression(length, order):
    """
    A simple linear autoregressive model with order "order",
    X_t = \sum_{i=0,...,order-1} Z_{t-i}
    Output is {X_t}_{t=0,...,length-1}
    """
    mu = 0.0
    sigma = 1.0
    Z1 = np.random.normal(mu, sigma, length + order)
    output1 = np.array(
        [np.sum(Z1[i: i + order]) / order for i in range(length)])

    Z2 = np.random.normal(mu, sigma, length + order)
    output2 = np.array(
        [np.sum(Z2[i: i + order]) / order for i in range(length)])

    output = np.transpose(np.array([output1, output2]))

    return output


def main():
    # Generate a two-dimensional timeseries
    timeseries = lin_autoregression(100000, 10)
    print(timeseries.shape)  # (100000, 2)
    print(covariance(timeseries))

    # Set parameters for the conf_int function
    number_bsamples = 1000
    alpha = 0.68
    stat = sbm.conf_int(
        timeseries, covariance, alpha, number_bsamples, parallel=True, method='bt')

    # Output the statistics
    print('Mean:', stat.mean, '\nStandard error:', stat.se, '\nLower limit:',
          stat.low, '\nUpper limit:', stat.up, '\nOptimal probability:', stat.prob)

    # Output the CDF of the bootstrap samples
    prob = np.arange(len(stat.dist)) / len(stat.dist)
    np.savetxt('dist.txt', np.transpose([stat.dist, prob]))


if __name__ == "__main__":
    main()
