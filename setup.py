#!/usr/bin/env python

from setuptools import setup

version = '1.1.0'

required = open('requirements.txt').read().split('\n')

setup(
    name='thunder-extraction',
    version=version,
    description='algorithms for feature extraction from spatio-temporal data',
    author='freeman-lab',
    author_email='the.freeman.lab@gmail.com',
    url='https://github.com/thunder-project/thunder-extraction',
    packages=['extraction', 'extraction.algorithms'],
    install_requires=required,
    long_description='See ' + 'https://github.com/thunder-project/thunder-extraction',
    license='MIT'
)
