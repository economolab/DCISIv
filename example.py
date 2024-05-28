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

# %% Data preparation

# Pre-computed PSTHs from (Inagaki et al., 2019), 50 ms bin size. Bin size 
# doesn't particularly matter for data with a trial structure to average 
# across. We recommend using the homogeneous model with total average firing 
# rates for data with no trial structure. See 
# https://en.wikipedia.org/wiki/Peristimulus_time_histogram for instructions on 
# how to calculate PSTHs

f_t = np.load('Inagaki2019_PSTHs.npy')
t = np.arange(0, 6, 0.05)

# Plotting all 755 PSTHs, vertical lines correspond to sample tone delivery, 
# start of delay period, and go cue delivery 
fig, ax = plt.subplots()
ax.plot(t, f_t.T)
plt.xlabel('t (s)')
plt.ylabel('f_t (Hz)')
plt.vlines([1.22, 2.39, 4.52], 0, ax.get_ylim()[1], colors='k', linestyles='dashed')

# Total average firing rate
f_t_homo = [np.mean(x) for x in f_t]

# Distribution of firing rates
fig, ax = plt.subplots()
plt.hist(f_t_homo, bins=20)
plt.xlabel('f_t (Hz)')
plt.ylabel('Count')

# ISI violation rate, pre-computed
ISI_v = np.load('Inagaki2019_ISI_v.npy')

# How to calculate ISI_v: 
# Assume vector of spike time occurences for a cluster "spks" 

# Recording time-referenced data:
# ISI_v = sum(np.diff(spks) < tau) / len(spks)

# Trial time-referenced data:
# n_viol = 0
# n_spks = 0
# for trial in trials:
#   n_viol += sum(np.diff(trial) < tau)
#   n_spks += len(trial)
# ISI_v = n_viol / n_spks 

# Distribution of ISI violation rates
fig, ax = plt.subplots()
plt.hist(f_t_homo, bins=20)
plt.xlabel('f_t (Hz)')
plt.ylabel('Count')

# %% Predicting FDR in real data

# Inhomogeneous model, default values other than censor period which, for this 
# dataset, appears to be 1 ms
res = DCISIv(f_t, ISI_v, tau_c=1).res

# printing results
print('\nMean FDR = {:.3f}'.format(res['mean']))
print('Median FDR = {:.3f}'.format(res['median']))
print(res['FDRs'])

# Homogeneous model, default values other than censor period which, for this 
# dataset, appears to be 1 ms
res = DCISIv(f_t_homo, ISI_v, tau_c=1).res

# printing results
print('\nMean FDR = {:.3f}'.format(res['mean']))
print('Median FDR = {:.3f}'.format(res['median']))
print(res['FDRs'])

# Use any arbitrary combination of assumed N
# res = DCISIv(f_t, ISI_v, N=1).res
# res = DCISIv(f_t, ISI_v, N=float('inf')).res
# res = DCISIv(f_t, ISI_v, N=(2, 3, 4)).res

# Use any arbitrary combination of asumed refractory or censor period 
# (both in ms)
# res = DCISIv(f_t, ISI_v, tau=1).res
# res = DCISIv(f_t, ISI_v, tau=2, tau_c=0.25).res

# %% Predicting FDR from simulated ISI_v 


# %% homogeneous firing sim vs analytic prediction, panel A, simulation

def economo_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + 0.5*(Rout**2)*2*tviol
    if Rin + Rout != 0:
        
        Fv = Rviol/(Rin + Rout)
    else:
        Fv = 0

    return Fv

def kleinfeld_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout
    if Rin + Rout != 0:
        
        Fv = Rviol/(Rin + Rout)
    else:
        Fv = 0

    return Fv


FDRs = [0.05, 0.20, 0.5]
Rtot = np.arange(1, 21, 1)

economo_sim_Fv = np.zeros((len(FDRs), len(Rtot)))
kleinfeld_sim_Fv = np.zeros((len(FDRs), len(Rtot)))

for i , val in enumerate(Rtot):
    for j, val2 in enumerate(FDRs):
        economo_sim_Fv[j, i] = neuronsim.sim_Fv_Fig1(val, FDR=val2, t_stop=1000)[0]
        kleinfeld_sim_Fv[j, i] = neuronsim.sim_Fv_Fig1(val, FDR=val2, t_stop=1000, 
                                                      out_refrac=2.5)[0]
        
# %%

res = DCISIv(Rtot, economo_sim_Fv[2,:], N=float('inf')).res

