# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 14:16:46 2023

@author: jpv88
"""

import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from matplotlib.path import Path
from matplotlib.patches import PathPatch
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

import JV_utils
import neuronsim



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

# %% predicted vs true across a range of conditions, panel D, simulation

from random import choices, sample, uniform
from scipy.optimize import minimize_scalar

from JV_utils import FDR_master

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

k = 1

N_con = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_con = np.array(choices(N_con, k=k), dtype='float')
N_con[N_con == 10] = float('inf')

Rtots = [4, 20]
Rtots = np.random.uniform(Rtots[0], Rtots[1], size=k)

FDRs = []
for _ in range(k):
    FDRs.append(uniform(0, 0.5))
    
PSTH_idx = list(range(len(PSTHs)))
idx_pairs = []
for _ in range(k):
    idx_pairs.append(sample(PSTH_idx, 2))

pred_FDR = []  

covs = []    
    
for i in range(k):
    
    Rin = PSTHs[idx_pairs[i][0]]
    Rout = PSTHs[idx_pairs[i][1]]
    Rin[Rin<0] = 0
    Rout[Rout<0] = 0
    
    Rout_target = FDRs[i]*Rtots[i]
    Rin_target = Rtots[i] - Rout_target
    
    scale = minimize_scalar(Rout_scale_ob, args=[Rin, Rin_target], 
                        method='bounded', bounds=[0, 100]).x
    Rin = Rin*scale

    scale = minimize_scalar(Rout_scale_ob, 
                        args=[Rout, (FDRs[i]/(1-FDRs[i]))*np.average(Rin)],
                        method='bounded', bounds=[0, 100]).x
    
    Rout = scale*Rout
    
    Rtot = Rin + Rout
    
    center = np.average(Rin)*np.average(Rout[0])
    covs.append(np.cov(Rin, Rout)[0,1])
    
    Fv, ISIs_full = neuronsim.sim_Fv_PSTH_gamma(Rin, 
                                                Rout, 
                                                neurons=N_con[i],
                                                cv=0.1,
                                                N=7200)
    ISIs_full = np.concatenate(ISIs_full)
    
    print(np.min(ISIs_full))
    
    pred_FDR.append(FDR_master(Fv, Rtot, Rout/np.linalg.norm(Rout), N_con[i]))
    
# %% test

from random import choices, sample, uniform
from scipy.optimize import minimize_scalar

from JV_utils import FDR_master

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

k = 1

N_con = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
N_con = np.array(choices(N_con, k=k), dtype='float')
N_con[N_con == 10] = float('inf')

Rtots = [4, 20]
Rtots = np.random.uniform(Rtots[0], Rtots[1], size=k)

FDRs = []
for _ in range(k):
    FDRs.append(uniform(0, 0.5))
    
PSTH_idx = list(range(len(PSTHs)))
idx_pairs = []
for _ in range(k):
    idx_pairs.append(sample(PSTH_idx, 2))

pred_FDR = []  

covs = []    
    
for i in range(k):
    
    Rin = PSTHs[idx_pairs[i][0]]
    Rout = PSTHs[idx_pairs[i][1]]
    Rin[Rin<0] = 0
    Rout[Rout<0] = 0
    
    Rout_target = FDRs[i]*Rtots[i]
    Rin_target = Rtots[i] - Rout_target
    
    scale = minimize_scalar(Rout_scale_ob, args=[Rin, Rin_target], 
                        method='bounded', bounds=[0, 100]).x
    Rin = Rin*scale

    scale = minimize_scalar(Rout_scale_ob, 
                        args=[Rout, (FDRs[i]/(1-FDRs[i]))*np.average(Rin)],
                        method='bounded', bounds=[0, 100]).x
    
    Rout = scale*Rout
    
    Rtot = Rin + Rout
    
    center = np.average(Rin)*np.average(Rout[0])
    covs.append(np.cov(Rin, Rout)[0,1])
    
    spks = neuronsim.sim_spikes_PSTH(Rin, Rout, out_refrac=2.5, N=30000)
    spks = np.concatenate(spks)
    R_actual = len(spks)/(7200*6)
    

    
    
# %%
sim = fullSim(pred_FDR, covs, FDRs, Rtots, N_con)
JV_utils.save_sim(sim, 'fullSim')

    
# %% plotting

import matplotlib as mpl

sim = JV_utils.load_sim('fullSim_07-12-2023_5')
pred_FDR = sim.pred_FDR
covs = sim.covs
FDRs = sim.FDRs
Rtots = sim.Rtots
N_con = sim.N_con

pred_FDR = np.array(pred_FDR)
FDRs = np.array(FDRs)

idxs = np.array(N_con) == 1
fig, ax = plt.subplots()
plt.scatter(FDRs, pred_FDR, c='blue', s=24)
plt.plot([0, 0.5], [0, 0.5], ls='dashed', c='k', lw=2)
plt.xlabel('True FDR', fontsize=16)
plt.ylabel('Predicted FDR', fontsize=16)
plt.text(0.3, 0.1, '$R^2$ = 0.98', fontsize=16)

y_pred, reg, R2 = JV_utils.lin_reg(FDRs, pred_FDR)

x = [0, 0.5]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*0.5 + reg.intercept_
y = [y1.item(), y2.item()]
plt.plot(x, y, c='k', lw=2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()


ax.set_aspect('equal')

ax.set_xlim(-0.025, 0.525)
ax.set_ylim(-0.025, 0.525)

plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'


