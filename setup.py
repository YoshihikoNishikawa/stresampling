from setuptools import setup, find_packages
import os

with open('README.md', 'r') as file:
    README = file.read()

setup_args = dict(
    name='stresampling',
    version="1.0.2",
    packages=['stresampling'],
    description="Python package for statistical analysis of time series using resampling methods",
    long_description=README,
    long_description_content_type="text/markdown",
    author='Yoshihiko Nishikawa, Jun Takahashi, and Takashi Takahashi',
    license='GPLv3',
    url='http://github.com/YoshihikoNishikawa/stresampling',
    python_requires='>=3.6',
    install_requires=['numpy>=1.13.3', 'scipy>=0.19.1', 'psutil>=5.9.0'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.6'
    ]
)


if __name__ == '__main__':
    setup(**setup_args)
