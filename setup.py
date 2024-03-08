# PSSL/setup.py

from setuptools import setup, find_packages

setup(
name = 'PSSL',
version ='0.1.0',
packages = find_packages(include=['core','core.*']),
description = 'Code for the paper "Probabilistic Self-Supervised Learning using Cyclical Stochastic Gradient MCMC"'
)