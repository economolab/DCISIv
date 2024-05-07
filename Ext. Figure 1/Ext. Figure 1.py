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
import gamma_spiking


# %%

class fullSim():
    
    def __init__(self, pred_FDR, FDRs, Rtots, N_con):
       
        self.pred_FDR = pred_FDR
        self.FDRs = FDRs
        self.Rtots = Rtots
        self.N_con = N_con


# %% predicted vs true across a range of conditions, panel D, simulation

from random import choices, sample, uniform
from scipy.optimize import minimize_scalar

from JV_utils import FDR_master

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

k = 100

N_con = [1, 2, 5, 10]
N_con = np.array(choices(N_con, k=k), dtype='float')
# N_con[N_con == 10] = float('inf')

Rtots = [4, 20]
Rtots = np.random.uniform(Rtots[0], Rtots[1], size=k)

FDRs = []
for _ in range(k):
    FDRs.append(uniform(0, 0.5))

pred_FDR = []    
Rtots_actual = [] 
    
for i in range(k):
    
    Fv, Rtot_actual = gamma_spiking.sim_Fv_lognormal(Rtots[i], 
                                                 FDRs[i], 
                                                 neurons=N_con[i],
                                                 T=7200,
                                                 CV=0.1,
                                                 N=1)
    Rtots_actual.append(Rtot_actual)
    
    Rtot_actual = [Rtot_actual] * 100
    Rout_hat = Rtot_actual/np.linalg.norm(Rtot_actual)
    pred_FDR_temp = FDR_master(Fv, Rtot_actual, Rout_hat, N_con[i])
    if pred_FDR_temp > 0.5:
        pred_FDR.append(0.5)
    else:
        pred_FDR.append(pred_FDR_temp)
    
    
# %%
sim = fullSim(pred_FDR, FDRs, Rtots_actual, N_con)
JV_utils.save_sim(sim, 'CV0.1_lognormal')

    
# %% plotting

import matplotlib as mpl

sim_name = 'CV0.1_lognormal_05-04-2024'

sim = JV_utils.load_sim(sim_name)
pred_FDR = sim.pred_FDR
FDRs = sim.FDRs
Rtots = sim.Rtots
N_con = sim.N_con
N_con[N_con == float('inf')] = 10

pred_FDR = np.array(pred_FDR)
FDRs = np.array(FDRs)

idxs = np.array(N_con) == 1
fig, ax = plt.subplots()
plt.scatter(FDRs, pred_FDR, c=N_con, s=24)
plt.plot([0, 0.5], [0, 0.5], ls='dashed', c='k', lw=2)
plt.xlabel('True FDR', fontsize=16)
plt.ylabel('Predicted FDR', fontsize=16)
plt.text(0.3, 0.1, '$R^2$ = 0.99', fontsize=16)
plt.title(sim_name, fontsize=16)

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

# %%

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

import gamma_spiking

rate = 10
refractory_period = 0.0025
CV = 2
t_stop = 2

spikes = gamma_spiking.gen_spikes_lognormal(CV, t_stop, rate, refractory_period)

ax.eventplot(spikes)

# %%

# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(1, 1)

# from scipy.stats import expon, gamma, invgauss, lognorm

# rate = 10
# refractory_period = 0.0025
# CV = 0.1

# def gamma_isi_gen(CV, refractory_period, rate):
    
#     shape_factor = 1/(CV**2)
#     scale = (1/(shape_factor*rate)) - (refractory_period/shape_factor)
#     isi_generator = gamma(a=shape_factor, scale=scale)
    
#     return isi_generator

# def invgauss_isi_gen(CV, refractory_period, rate):
    
#     nu = (1-refractory_period*rate)/rate
#     lam = nu/(CV**2)
#     isi_generator = invgauss(mu=nu/lam, loc=refractory_period, scale=lam)
    
#     return isi_generator

# def lognorm_isi_gen(CV, refractory_period, rate):
    
#     sigma = np.sqrt(np.log(CV**2 + 1))
#     mu = np.log((1/rate) - refractory_period) - (sigma**2)/2
#     isi_generator = lognorm(s=sigma, scale=np.exp(mu))
    
#     return isi_generator

# isi_generator = gamma_isi_gen(CV, refractory_period, rate)

# x = np.linspace(isi_generator.ppf(0.0001),
#                 isi_generator.ppf(0.9999), 1000)

# ax.plot(x, isi_generator.pdf(x),
#        'r-', lw=2, alpha=0.6)

# plt.show()


