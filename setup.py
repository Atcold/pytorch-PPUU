#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='ppuu',
      version='0.0.1',
      description='Predictive Policy Under Uncertainty',
      author='',
      author_email='',
      url='https://github.com/vladisai/pytorch-PPUU',
      install_requires=[
          'pytorch-lightning',
          'torch',
          'numpy',
          'pandas',
      ],
      packages=find_packages()
      )
