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
import SpikeSim

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
plt.hist(ISI_v, bins=20)
plt.xlabel('ISI_v')
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

# %% Predicting FDR from simulated ISI_v (homo)

f_t = 8
FDR = 0.15
N = 2
t_stop = 1000

ISI_v = SpikeSim.sim_ISI_v_homo(f_t, FDR, N=N, t_stop=t_stop)
pred_FDR = DCISIv(f_t, ISI_v, N=N).res['mean']

print('\nObserved ISI_v = {:.3f}'.format(ISI_v))
print('Predicted FDR = {:.3f}'.format(pred_FDR))
print('True FDR = {:.3f}'.format(FDR))

# %% Predicting FDR from simulated ISI_v (inhomo)

f_t_avg = 8
f_TP_idx = 0
f_FP_idx = 1
FDR = 0.15
N = 1
t_stop = 1000

f_TP = f_t[f_TP_idx,:]
f_FP = f_t[f_FP_idx,:]

def PSTH_scale_ob(scale, args):
    PSTH, PSTH_avg_target = args
    return abs(np.average(scale*PSTH) - PSTH_avg_target)

f_FP_avg = f_t_avg * FDR
f_TP_avg = f_t_avg - f_FP_avg

f_TP = f_TP / np.linalg.norm(f_TP)
f_FP = f_FP / np.linalg.norm(f_FP)

scale = minimize_scalar(PSTH_scale_ob, args=[f_TP, f_TP_avg], 
                    method='bounded', bounds=[0, 100]).x
f_TP = f_TP*scale

scale = minimize_scalar(PSTH_scale_ob, 
                    args=[f_FP, f_FP_avg],
                    method='bounded', bounds=[0, 100]).x

f_FP = scale * f_FP

ISI_v = SpikeSim.sim_ISI_v_inhomo(f_t_avg, FDR, f_TP, f_FP, N=N, t_stop=t_stop)
pred_FDR = DCISIv(np.vstack([f_TP, f_FP]), ISI_v, N=N).res['mean']

print('\nObserved ISI_v = {:.3f}'.format(ISI_v))
print('Predicted FDR = {:.3f}'.format(pred_FDR))
print('True FDR = {:.3f}'.format(FDR))


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

