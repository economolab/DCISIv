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

# %%
class HomoHeteroSim():
    
    def __init__(self, Rin, pred_FDR, Rin_out_dot, Rtot_out_dot, Fvs, R_out_mags,
                 R_out_units, pred_FDR_true):
       
        self.Rin = Rin
        self.pred_FDR = pred_FDR
        self.Rin_out_dot = Rin_out_dot
        self.Rtot_out_dot = Rtot_out_dot
        self.Fvs = Fvs
        self.R_out_mags = R_out_mags
        self.R_out_units = R_out_units
        self.pred_FDR_true = pred_FDR_true
        
# %%
class fullSim():
    
    def __init__(self, pred_FDR, covs, FDRs, Rtots, N_con):
       
        self.pred_FDR = pred_FDR
        self.covs = covs
        self.FDRs = FDRs
        self.Rtots = Rtots
        self.N_con = N_con

# %% homogeneous firing sim vs analytic prediction, panel B, simulation

def economo_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout + 0.5*(Rout**2)*2*tviol
    Fv = Rviol/(Rin + Rout)

    return Fv

def kleinfeld_Fv(Rin, Rout, tviol=0.0025):

    Rviol = 2*tviol*Rin*Rout
    Fv = Rviol/(Rin + Rout)

    return Fv

Rtot = [1, 8, 20]
FDRs = np.arange(0, 1.1, 0.1)

economo_sim_Fv = np.zeros((len(FDRs), len(Rtot)))
kleinfeld_sim_Fv = np.zeros((len(FDRs), len(Rtot)))

for i, val in enumerate(FDRs):
    for j, val2 in enumerate(Rtot):
        economo_sim_Fv[i,j] = neuronsim.sim_Fv_Fig1(val2, FDR=val, 
                                                    t_stop=1000)[0]
        kleinfeld_sim_Fv[i,j] = neuronsim.sim_Fv_Fig1(val2, FDR=val, 
                                                      t_stop=1000, 
                                                      out_refrac=2.5)[0]


FDRs_2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.66]
sim_Fv_N_2 = np.zeros((len(FDRs_2), len(Rtot)))
for i, val in enumerate(FDRs_2):
    for j, val2 in enumerate(Rtot):
        sim_Fv_N_2[i,j] = neuronsim.sim_Fv_neurons(val2, FDR=val, neurons=2, 
                                                 t_stop=1000)[0]
        
FDRs_5 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.83]
sim_Fv_N_5 = np.zeros((len(FDRs_5), len(Rtot)))
for i, val in enumerate(FDRs_5):
    for j, val2 in enumerate(Rtot):
        sim_Fv_N_5[i,j] = neuronsim.sim_Fv_neurons(val2, FDR=val, neurons=5, 
                                                 t_stop=1000)[0]
    
    
# %% plotting

fig, ax = plt.subplots()

from matplotlib import cm

start = 0.0
stop = 1.0
number_of_lines= 3
cm_subsection = np.linspace(start, stop, number_of_lines)

colors = [ cm.copper(x) for x in cm_subsection ]

colors = ['purple', 'green', 'orange']
colors_scat = ['darkmagenta', 'darkgreen', 'darkorange']
cmaps = ['Purples', 'Greens', 'Oranges']

for idx in range(0, 3):

    R_out = FDRs*Rtot[idx]
    R_in = Rtot[idx] - R_out
    
    plt.scatter(FDRs, np.array(economo_sim_Fv[:,idx])*100, s=20, c=colors_scat[idx])
    plt.scatter(FDRs[:6], np.array(kleinfeld_sim_Fv[:6,idx])*100, s=20, c=colors_scat[idx])
    plt.plot(FDRs, economo_Fv(R_in, R_out)*100, lw=3, c=colors[idx], label='N = ∞')
    plt.plot(FDRs[:6], kleinfeld_Fv(R_in[:6], R_out[:6])*100, lw=3, 
             c=colors[idx], label='N = 1', ls='dashed')
    # plt.scatter(FDR_N, np.array(sim_Fv_N[:,idx])*100, s=20, c='black')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.set_ylim(-0.2, 5.2)
    # bottom_line = list(kleinfeld_Fv(R_in[:6], R_out[:6])*100) + \
        # list(np.array(sim_Fv_N[:,idx])*100) + [economo_Fv(R_in[-1], R_out[-1])*100]
    # plt.plot(FDRs[5:], bottom_line[5:], lw=3, c='black')
    # ax.fill_between(FDRs, economo_Fv(R_in, R_out)*100, bottom_line, 
                    # facecolor="none", hatch="+", edgecolor="k", linewidth=0.0, 
                    # alpha=0.2)

    
    path = Path(np.vstack((np.vstack((FDRs[:6], kleinfeld_Fv(R_in[:6], R_out[:6])*100)).T, 
                       [FDRs[5], kleinfeld_Fv(R_in[5], R_out[5])*100], 
                       [FDRs[-1], economo_Fv(R_in[-1], R_out[-1])*100],
                       np.flip(np.vstack((FDRs, economo_Fv(R_in, R_out)*100)).T, axis=0))))
    patch = PathPatch(path, facecolor ='none', edgecolor='none')
    
    ax.add_patch(patch)
    # im = ax.imshow([[0.,1.], [0.,1.]], interpolation ='bilinear', cmap = plt.cm.gray,
    #                clip_path = patch, clip_on = True)
    
    lims = [ax.get_xlim(), ax.get_ylim()]
    lims = [item for t in lims for item in t]
    im = ax.imshow([[0.,0.], [1.,0.]], interpolation ='bicubic', cmap = cmaps[idx],
                   extent=lims, aspect='auto', clip_path = patch, clip_on = True,
                   zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xlabel('FDR', fontsize=16)
# plt.title('$R_{tot}$ = 16 Hz', fontsize=18)
plt.ylabel('% $ISI_{viol}$', fontsize=16)
# plt.legend(prop={'size': 16})

for col in sim_Fv_N_2.T:
    plt.plot(FDRs_2, col*100, c='black', alpha=0.4, ls='dotted', zorder=1)
for col in sim_Fv_N_5.T:
    plt.plot(FDRs_5, col*100, c='black', alpha=0.4, ls='dotted', zorder=1)
    
# plt.text(1.05, 0.15, '1 Hz', fontsize='16')
# plt.text(0.8, 1.4, '10 Hz', fontsize='16')
# plt.text(0.8, 3.5, '20 Hz', fontsize='16')
# plt.text(1.05, 0.15, '1 Hz', fontsize='16')
# plt.text(1.05, 0.95, '4 Hz', fontsize='16')
# plt.text(1.05, 3.9, '16 Hz', fontsize='16')

from matplotlib.lines import Line2D
custom_lines = [Line2D([0], [0], c='k'),
                Line2D([0], [0], c='k', ls='dotted'),
                Line2D([0], [0], c='k', ls='dashed')]
ax.legend(custom_lines, ['N = ∞', 'N = 2/5', 'N = 1'], prop={'size': 16})

import matplotlib as mpl

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.tight_layout()

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
Rtot = np.arange(0, 21, 1)

economo_sim_Fv = np.zeros((len(FDRs), len(Rtot)))
kleinfeld_sim_Fv = np.zeros((len(FDRs), len(Rtot)))

for i , val in enumerate(Rtot):
    for j, val2 in enumerate(FDRs):
        economo_sim_Fv[j, i] = neuronsim.sim_Fv_Fig1(val, FDR=val2, t_stop=1000)[0]
        kleinfeld_sim_Fv[j, i] = neuronsim.sim_Fv_Fig1(val, FDR=val2, t_stop=1000, 
                                                      out_refrac=2.5)[0]
        
# %% plotting

F_v_economo = np.zeros((len(FDRs), len(Rtot)))
F_v_kleinfeld = np.zeros((len(FDRs), len(Rtot)))

for i, val in enumerate(FDRs):
    R_out = val*Rtot
    R_in = Rtot - R_out
    for j, val2 in enumerate(range(len(R_out))):
        F_v_economo[i, j] = economo_Fv(R_in[j], R_out[j])*100
        F_v_kleinfeld[i, j] = kleinfeld_Fv(R_in[j], R_out[j])*100
        
from matplotlib import cm

start = 0.0
stop = 1.0
number_of_lines= 3
cm_subsection = np.linspace(start, stop, number_of_lines)

c1, c2, c3 = [ cm.magma(x) for x in cm_subsection ]
 
fig, ax1 = plt.subplots()
plt.scatter(Rtot, economo_sim_Fv[0,:]*100, s=20, c='darkblue')
plt.scatter(Rtot, kleinfeld_sim_Fv[0,:]*100, s=20, c='darkblue')
plt.scatter(Rtot, economo_sim_Fv[1,:]*100, s=20, c='darkgreen')
plt.scatter(Rtot, kleinfeld_sim_Fv[1,:]*100, s=20, c='darkgreen')
plt.scatter(Rtot, economo_sim_Fv[2,:]*100, s=20, c='darkred')
plt.scatter(Rtot, kleinfeld_sim_Fv[2,:]*100, s=20, c='darkred')
plt.plot(Rtot, F_v_economo[0,:], lw=3, c='blue')
plt.plot(Rtot, F_v_kleinfeld[0,:], lw=3, c='blue', ls='dashed')
plt.plot(Rtot, F_v_economo[1,:], lw=3, c='green')
plt.plot(Rtot, F_v_kleinfeld[1,:], lw=3, c='green', ls='dashed')
plt.plot(Rtot, F_v_economo[2,:], lw=3, c='red')
plt.plot(Rtot, F_v_kleinfeld[2,:], lw=3, c='red', ls='dashed')

# ax1.fill_between(Rtot, F_v_economo[0,:], F_v_kleinfeld[0,:], facecolor="none", 
#                 hatch="+", edgecolor="k", linewidth=0.0, alpha=0.2)
# ax1.fill_between(Rtot, F_v_economo[1,:], F_v_kleinfeld[1,:], facecolor="none", 
#                 hatch="+", edgecolor="k", linewidth=0.0, alpha=0.2)
# ax1.fill_between(Rtot, F_v_economo[2,:], F_v_kleinfeld[2,:], facecolor="none", 
#                 hatch="+", edgecolor="k", linewidth=0.0, alpha=0.2)
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
ax2 = ax1.twiny() # ax1 and ax2 share y-axis
ax3 = ax1.twinx()
ax2.set_xticks([0, 1, 2, 3, 4 ,5])
ax1.set_xticks([0, 4, 8, 12, 16, 20])
ax2.set_xticklabels([0, 1, 2, 3, 4 ,5], fontsize=14)
ax1.set_xticklabels([0, 4, 8, 12, 16, 20], fontsize=14)
ax2.set_xlim(-0.25, 5.25)
ax1.set_ylim(-0.1875, 4.1875)
ax1.set_xlabel('$R_{tot}$ (Hz)', fontsize=16)
ax1.set_ylabel('% $ISI_{viol}$', fontsize=16)
# plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
left_ticks= [0, 1, 2, 3, 4]
ax1.set_yticks(left_ticks)
ax1.set_yticklabels(left_ticks, fontsize=14)
plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)
ax3.set_ylim(0-(3/64), 1+(3/64))
right_ticks = [0.00, 0.25, 0.50, 0.75, 1.00]
ax3.set_yticks(right_ticks)
ax3.set_yticklabels(right_ticks, fontsize=14)

alt_ax_color = 'orange'
ax1.spines['right'].set_color(alt_ax_color)
ax2.spines['right'].set_color(alt_ax_color)
ax3.spines['right'].set_color(alt_ax_color)
ax2.spines['top'].set_color(alt_ax_color)
ax1.spines['top'].set_color(alt_ax_color)
ax3.spines['top'].set_color(alt_ax_color)
ax2.tick_params(axis='x', colors=alt_ax_color)
ax3.tick_params(axis='y', colors=alt_ax_color)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax3.spines['top'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)

cmaps = ['Blues', 'Greens', 'Reds']
for i in range(3):
    up_kleinfeld = np.vstack((Rtot, F_v_kleinfeld[i,:])).T
    down_economo = np.flip(np.vstack((Rtot, F_v_economo[i,:])).T, axis=0)
    straight_up = np.vstack((up_kleinfeld[-1], down_economo[0]))
    path = Path(np.vstack((up_kleinfeld, straight_up, down_economo)))
    patch = PathPatch(path, facecolor ='none', edgecolor='none')
    
    ax1.add_patch(patch)
    # im = ax.imshow([[0.,1.], [0.,1.]], interpolation ='bilinear', cmap = plt.cm.gray,
    #                clip_path = patch, clip_on = True)
    
    lims = [ax1.get_xlim(), ax1.get_ylim()]
    lims = [item for t in lims for item in t]
    im = ax1.imshow([[0.,0.], [1.,0.]], interpolation ='bicubic', cmap = cmaps[i],
                   extent=lims, aspect='auto', clip_path = patch, clip_on = True)

#plt.xscale('log')
ax1.grid(axis='both', which='both', ls ='--', alpha=0.4)
# from matplotlib.lines import Line2D
# custom_lines = [Line2D([0], [0], c='k'),
#                 Line2D([0], [0], c='k', ls='dashed')]
# # custom_lines.append('FDR')
# L = ax1.legend(custom_lines, ['N = ∞', 'N = 1'], prop={'size': 16})
# plt.text(21, 3, '50%', fontsize='16')
# plt.text(21, 1.6, '20%', fontsize='16')
# plt.text(21, 0.4, '5%', fontsize='16')
#plt.text(5.1, 0.75, '50%', fontsize='16')
#plt.text(5.1, 0.4, '20%', fontsize='16')
#plt.text(5.1, 0.1, '5%', fontsize='16')
#plt.xlim(-1.0, 21.0)
# plt.ylim(0, 1)
import matplotlib as mpl

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

plt.tight_layout()

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

# %% heterogeneous vs homogeneous predictions of heterogeneous simulations, 
# panel C, simulation

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

from scipy.optimize import minimize_scalar

def vector_mag(data):
    sum_squares = 0
    for el in data:
        sum_squares += el**2
    return (sum_squares)**(1/2)

# the final equation
def FDR_master(ISIviol, Rtot, Rout_unit, N, tau=2.5, tau_c=0):
    
    Rtot_mag = np.linalg.norm(Rtot)
    Rtot_avg = np.average(Rtot)
    Rtot_unit = Rtot/Rtot_mag
    
    tau = tau*(10**-3)
    n = len(Rtot)
    D = np.dot(Rtot_unit, Rout_unit)

    if N == float('inf'):
        N_correction1 = 1
        N_correction2 = 1
    else:
        N_correction1 = N/(N + 1)
        N_correction2 = (N + 1)/N
    
    sqrt1 = (Rtot_mag**2)*(D**2)
    sqrt2 = (N_correction2*Rtot_avg*ISIviol*n)/(tau - tau_c)
    Rout_mag = N_correction1*(Rtot_mag*D - (sqrt1 - sqrt2)**(0.5))
    
    Rout_avg = np.average(Rout_mag*np.array(Rout_unit))
    FDR = Rout_avg/Rtot_avg
    
    return FDR

pred_FDR = []
Rtots = []
Rin_out_dot = []
Rtot_out_dot = []
Fvs = []
R_out_mags = []
R_out_units = []
pred_FDR_true = []

# ideal idx is 693
main_idx = 693
other_idx = list(range(len(PSTHs)))
del other_idx[main_idx]

other_idx = other_idx[:100]

idx1 = main_idx
Rin = PSTHs[idx1]

scale = minimize_scalar(Rout_scale_ob, args=[Rin, 10], 
                        method='bounded', bounds=[0, 100]).x
Rin = Rin*scale

for second_idx in other_idx:
    print(second_idx)

    idx2 = second_idx
    
    FDR = 0.2
    
    scale = minimize_scalar(Rout_scale_ob, 
                            args=[PSTHs[idx2], (FDR/(1-FDR))*np.average(Rin)],
                            method='bounded', bounds=[0, 100]).x
    
    Rout = scale*PSTHs[idx2]
    
    Fv_temp = neuronsim.sim_Fv_PSTH3(Rin, Rout, FDR=FDR, out_refrac=2.5, 
                                     N=10000)
    
    R_out_mags.append(vector_mag(Rout))
    Rtot =  Rin + Rout
    R_out_units.append(Rout/vector_mag(Rout))
    
    Rin_out_dot.append(np.dot(Rin, Rout))
    Rtot_out_dot.append(np.dot(Rtot, Rout))
    
    Fv = Fv_temp
    Fvs.append(Fv)
    D = np.dot(Rtot/vector_mag(Rtot), Rout/vector_mag(Rout))
    
    pred_FDR.append(FDR_master(Fv, [np.average(Rtot)]*1000, 
                               ([np.mean(Rout)]*1000)/vector_mag([np.mean(Rout)]*1000), 
                               1))
    
    pred_FDR_true.append(FDR_master(Fv, Rtot, Rout/vector_mag(Rout), 1))
    

sim = HomoHeteroSim(Rin, pred_FDR, Rin_out_dot, Rtot_out_dot, Fvs, R_out_mags, 
                    R_out_units, pred_FDR_true)
    
JV_utils.save_sim(sim, 'HomovsHetero')
    
    
# %% plotting

sim = JV_utils.load_sim('HomovsHetero_07-07-2023_2')
Rin = sim.Rin
pred_FDR = sim.pred_FDR
Rin_out_dot = sim.Rin_out_dot
Rtot_out_dot = sim.Rtot_out_dot
Fvs = sim.Fvs
R_out_mags = sim.R_out_mags
R_out_units = sim.R_out_units
pred_FDR_true = sim.pred_FDR_true

pred_FDR_true = np.array(pred_FDR_true)

fig, ax = plt.subplots()

Rout = []
for i in range(len(R_out_mags)):
    
    Rout.append(R_out_mags[i]*R_out_units[i])

covs = []    
for i in range(len(Rout)):
    covs.append(np.cov(Rin, Rout[i])[0,1])

center = np.average(Rin)*np.average(Rout[0])
plt.scatter(covs, pred_FDR, s=20, c='g')

x = np.array(Rin_out_dot)[np.invert(np.isnan(pred_FDR))]
y = np.array(pred_FDR)[np.invert(np.isnan(pred_FDR))]

y_pred, reg, R2 = JV_utils.lin_reg(x, y)
temp = (x/1000 - center)/center
xs = covs
ys = y_pred
ys = JV_utils.sort_list_by_list(xs, ys)
xs = sorted(xs)
xs = [xs[0], xs[-1]]
ys = [ys[0], ys[-1]]

plt.plot(xs, ys, c='g', ls='dotted')
ax.axvline(0, c='k', ls='--', alpha=0.3)
ax.axhline(0.2, c='k', ls='--', alpha=0.3)
plt.ylabel('Predicted FDR', fontsize=16)
plt.xlabel(r'$\overline{R_{in}R_{out}}$ (ẟ)', 
           fontsize=16)
plt.scatter(covs, pred_FDR_true[~np.isnan(pred_FDR)], 
            s=10, marker='x', zorder=0, c='b')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
# plt.text(0.6, 0.12,'$R^{}$ = {}'.format(2, str(round(R2, 2))), fontsize=16)
# plt.title('$Unit_{idx}$ = 60', fontsize=18)

import matplotlib as mpl

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.tight_layout()

# %% sample plots of extreme, minimal, and medium overlap for x-axis

def closest_value(input_list, input_value):

  arr = np.asarray(input_list)
  i = (np.abs(arr - input_value)).argmin()

  return i, arr[i]

overlap_factor = (np.array(Rin_out_dot)/1000 - center)/center

print(closest_value(overlap_factor, 0))

# idxs to use: 70, 9, 17
idx1 = 693
idx2 = 17

fig, ax = plt.subplots()
plt.plot(PSTHs[idx1]/vector_mag(PSTHs[idx1]))
plt.plot(PSTHs[idx2]/vector_mag(PSTHs[idx2]))

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.tight_layout()

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

k = 100

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
    
    Fv = neuronsim.sim_Fv_PSTH4(Rin, Rout, out_refrac=2.5, 
                                neurons=N_con[i], N=7200)
    
    
    pred_FDR.append(FDR_master(Fv, Rtot, Rout/np.linalg.norm(Rout), N_con[i]))
    
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
