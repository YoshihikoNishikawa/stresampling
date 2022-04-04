

# **stresampling**

![license](https://img.shields.io/badge/license-GPLv3-brightgreen)
[![pypi](https://img.shields.io/pypi/v/stresampling)](https://pypi.python.org/pypi/stresampling)
[![python](https://img.shields.io/pypi/pyversions/stresampling)](https://pypi.python.org/pypi/stresampling)
## The aim of the package

The Python package **stresampling** implements resampling methods applicable to stationary timeseries, especially the stationary bootstrap ([Politis, D. N. and Romano, J. P.](https://www.jstor.org/stable/2290993)) method for estimating statistical properties of stationary timeseries, using 
[the bootstrap percentile](https://www.taylorfrancis.com/books/mono/10.1201/9780429246593/introduction-bootstrap-bradley-efron-tibshirani), 
[the bias-corrected](https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-confidence-intervals/10.1214/ss/1032280214.full), 
or [the bootstrap-t](https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-confidence-intervals/10.1214/ss/1032280214.full) methods, 
with an optimal choice of the parameter 
([Politis, D.N. and White, H.](http://www.tandfonline.com/doi/abs/10.1081/ETC-120028836), [Patton, A., Politis, D.N. and White, H.](http://www.tandfonline.com/doi/abs/10.1080/07474930802459016)) from a single time series. 

The aim of the package is to provide an easy and intuitive interface to calculate statistics of desired quantities using single one timeseries, with minimal dependencies on other packages.

## Authors
[Yoshihiko Nishikawa (Tohoku University)](mailto:yoshihiko.nishikawa.a7@tohoku.ac.jp), [Jun Takahashi (University of New Mexico)](https://github.com/JunGitef17), and [Takashi Takahashi (University of Tokyo)](https://github.com/takashi-takahashi)



## Requirements
- Python>=3.6
- numpy>=1.13.3
- scipy>=0.19.1
- psutil>=5.9.0


## Usage
The **stresampling** package is very simple and intuitive to use. What you need are the timeseries you wish to analyze, a quantity of interest, and the coverage probability for the output confidence interval.

The following is a simple example showing how to use the package:
```python
import numpy as np
from stresampling import stationary_bootstrap as sbm

def phys(timeseries): # Kurtosis of the distribution
    return np.mean(timeseries**4.0) / np.mean(timeseries**2.0)**2.0

def main():
    file = 'data/timeseries.dat'
    timeseries = np.loadtxt(file)
    print(timeseries.shape) # (10000,)
    print(timeseries)

    alpha = 0.68
    stat = sbm.conf_int(timeseries, phys, alpha)
    print('Mean:', stat.mean, 'Standard error:', stat.se, 'Lower and upper confidence limits:', stat.low, stat.up)

if __name__ == "__main__":
    main()
```


## Detail of the `conf_int` function

<pre>
stresampling.stationary_bootstrap.conf_int(<i>seq</i>, <i>phys</i>, <i>alpha</i>, <i>number_bsamples</i>, <i>parallel=True</i>, <i>method='percentile'</i>)
</pre>

- parameters

    - `seq`: ndarray
        
        Input timeseries of shape (timeseries length, *)

    - `phys`: Function
    
        A method to calculate the desired quantity

        Note that the `conf_int` function currently supports only a scalar output

    - `alpha`: Real number in [0, 1]

        The coverage probability of the output confidence interval

    - `number_bsamples`: Integer value, *optional*

        The number of bootstrap samples to be built

        The default value is 1000

    - `parallel`: Bool, *optional*

        Build bootstrap samples using multiple cores if True

        The default is True

    - `method`: {'percentile', 'bt', 'symbt', 'bc'}, *optional*

        Specify which method will be used to estimate the confidence limits

        The default is 'percentile'

        - 'percentile': [the bootstrap percentile method](https://www.taylorfrancis.com/books/mono/10.1201/9780429246593/)
        - 'bt': [the bootstrap-t method](https://projecteuclid.org/journals/annals-of-statistics/volume-24/issue-5/Second-order-correctness-of-the-blockwise-bootstrap-for-stationary-observations/10.1214/aos/1069362303.full)
        - 'symbt': [the symmetric bootstrap-t method](https://www.tandfonline.com/doi/abs/10.1080/10485250600687812)
        - 'bc': [the bias-corrected method](https://projecteuclid.org/journals/statistical-science/volume-11/issue-3/Bootstrap-confidence-intervals/10.1214/ss/1032280214.full). Note that no acceleration is used in this package.

- Return: Stat: a class including
    - `mean`: The estimate of the quantity
    - `se`: The standard error of the estimate
    - `low` and `up`: The lower and upper confidence limits
    - `prob`: The estimated optimal probability for the stationary bootstrap method 
    - `dist`: *ndarray* of the sorted bootstrap samples of the quantity
    - `autocorr`: *ndarray* for the unnormalized autocorrelation function of the timeseries

## Installation

You can easily install the package via pypi as
```shell
pip install stresampling
```
or, by cloning the repository,
```shell
git clone https://github.com/YoshihikoNishikawa/stresampling.git
cd stresampling
pip install .
```

## Future development

We will implement other resampling methods such as [the circular bootstrap](https://www.wiley.com/Exploring+the+Limits+of+Bootstrap-p-9780471536314) and [the subsampling method](https://projecteuclid.org/journals/annals-of-statistics/volume-22/issue-4/Large-Sample-Confidence-Regions-Based-on-Subsamples-under-Minimal-Assumptions/10.1214/aos/1176325770.full) in the future updates.

## Citation

If you use this package or find it useful, please cite [arXiv:2112.11837](https://arxiv.org/abs/2112.11837).

## Contributing
If you wish to contribute, please submit a pull request.

If you find an issue or a bug, please contact us or raise an issue. 
