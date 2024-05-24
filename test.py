# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:16:12 2024

@author: jpv88
"""

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import JV_utils
import neuronsim

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from random import choices, sample
from scipy.optimize import minimize_scalar
from scipy.stats import cauchy
from scipy.interpolate import splev, splrep
from sklearn.metrics import r2_score
from scipy import stats
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d


from DCISIv import DCISIv

# %% load in and preprocess PSTHs

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']

t = np.linspace(0, 6, 120)
t_new = np.linspace(0, 6, 1000)

PSTHs_new = []
for i in range(len(PSTHs)):
    smoothed = gaussian_filter1d(PSTHs[i], 3)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    PSTHs_new.append(y_new)
    
PSTHs = np.vstack(PSTHs_new)
ISI_v = (np.repeat(0.01, 755))

PSTHs_homo = [np.mean(x) for x in PSTHs_new]

# %%

res = DCISIv(PSTHs_homo, ISI_v, N=(2,3,4,5)).res

