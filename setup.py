# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:00:28 2024

@author: jpv88
"""

from setuptools import setup

setup(
    name='DCISIv',
    version='0.1',    
    description='Decoding Cross-contamination using Inter-Spike-Interval violations',
    url='https://github.com/economolab/DCISIv',
    author='Jack Vincent',
    author_email='jackv@bu.edu',
    license='MIT',
    packages=['DCISIv'],
    install_requires=['warnings',
                      'numpy',
                      'pandas',
                      'random',
                      'tqdm',
                      'os',
                      'matplotlib',
                      'scipy',
                      'elephant',
                      'neo',
                      'quantities'
                      ],

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',          
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
        'Programming Language :: Python :: 3.14' 
    ],
)