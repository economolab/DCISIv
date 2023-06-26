# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:55:12 2023

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
import pandas as pd
import scipy.io as sio

from numpy import linalg as LA
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

import JV_utils
import neuronsim

# %% funcs 

# calculate overall ISI violation fraction given a list of lists corresponding 
# to spike times in multiple trials
def Fv_calc(trials):
    
    viols = 0
    spks = 0
    for trial in trials:
        viols += sum(np.diff(trial) < 0.0025)
        spks += len(trial)
    
    return viols/spks

# produces mirrored ISI distribution given a number of simulation parameters, 
# also returns overall ISI violation fraction and total firing rate
def gen_ISIs(Rin, Rout, N, t_stop, out_refrac):
    
    spks = neuronsim.sim_spikes(Rin, Rout, N=N, t_stop=t_stop, out_refrac=out_refrac)[0]
    Fv = Fv_calc(spks)
    Rtot = len(np.concatenate(spks))/(t_stop*N)
    
    ISIs = []
    for trial in spks:
        ISIs.append(np.diff(trial))
    ISIs = np.concatenate(ISIs)
    ISIs = ISIs*1000
    ISIs = np.concatenate((ISIs, ISIs*-1))
    
    return ISIs, Fv, Rtot

# plot ISI histogram for a given ISI distribution in specific subplot axes, i 
# is row index, j is column index
def plot_ISIs(ISIs_df, k, ax, i, j):
    
    ISIs = ISIs_df['ISIs'][k]
    Fv = ISIs_df['Fv'][k]
    
    bins = np.arange(-25.5, 25.5, 0.5)
    ax[i,j].hist(ISIs, bins, facecolor='blue', edgecolor='black', zorder=3)
    plt.grid(axis='y', which='both', ls ='--', alpha=0.3, lw=0.1)
    
    ax[i,j].set_title('$ISI_{viol} = $' + str(round(Fv*100, 1)) + '%', 
                      fontsize=18)
    ax[i,j].set_xticks([-20, 20])
    for tick in ax[i,j].xaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    
    y_max = max(np.histogram(ISIs, bins)[0])
    ax[i,j].set_yticks([0, y_max])
    for tick in ax[i,j].yaxis.get_major_ticks():
        tick.label.set_fontsize(16)
    fig.patches.extend([plt.Rectangle((-2.5, 0), 5, ax[i,j].get_ylim()[1],
                                      color='r', alpha=0.2, zorder=1000, 
                                      fill=True, transform=ax[i,j].transData)])
    
    
# %% Changing firing rate

ISIs_df = pd.DataFrame(columns=['ISIs', 'Fv', 'Rtot'])

Rin = [19, 0.8*5.9375, 1.9]
Rout = [1, 0.2*5.9375, 1.9]

for i in range(3):
    
    ISIs, Fv, Rtot = gen_ISIs(Rin[i], Rout[i], 100, 6, 2.5)
    ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]

# %% changing PSTH overlap

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']

t = np.linspace(0, 6, 120)
t_new = np.linspace(0, 6, 1000)

def preprocess(PSTH, sigma):
    unit = PSTH/LA.norm(PSTH)
    smoothed = gaussian_filter1d(unit, sigma)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    return y_new


true1 = PSTHs[687]
rogue_base = true1 + np.random.normal(0, 1, [len(true1),])
rogue1 = np.roll(rogue_base, 50)
rogue2 = np.roll(rogue_base, 100)
rogue3 = np.roll(rogue_base, 120)

sigma = 5
true1 = preprocess(true1, sigma)
rogue1 = preprocess(rogue1, sigma)
rogue2 = preprocess(rogue2, sigma)
rogue3 = preprocess(rogue3, sigma)

true1[true1<0] = 0
rogue1[rogue1<0] = 0
rogue2[rogue2<0] = 0
rogue3[rogue3<0] = 0

rogues = [rogue1, rogue2, rogue3]
true_scale = 100
rogue_scale = 40

for i in range(3):
    spks = neuronsim.sim_spikes_PSTH2(true1*true_scale, rogues[i]*rogue_scale, N=100, out_refrac=2.5)
    
    Fv = Fv_calc(spks)
    Rtot = len(np.concatenate(spks))/(100*6)

    ISIs = []
    for trial in spks:
        ISIs.append(np.diff(trial))
    ISIs = np.concatenate(ISIs)
    ISIs = ISIs*1000
    ISIs = np.concatenate((ISIs, ISIs*-1))
    print(Fv)

    ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]
    
# %% generate PSTH traces

fig, ax = plt.subplots()
plt.plot(true1*100)
plt.plot(rogue1*50)

fig, ax = plt.subplots()
plt.plot(true1*100)
plt.plot(rogue2*50)

fig, ax = plt.subplots()
plt.plot(true1*100)
plt.plot(rogue3*50)
    
# %% changing confounding neuron count

ISIs, Fv, Rtot = gen_ISIs(0.7071067811865476*20, 0.2928932188134524*20, 100, 6, 0)
ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]

ISIs, Fv, Rtot = gen_ISIs(10, 10, 100, 6, 2.5)
ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]
    
spks = neuronsim.sim_Fv_neurons(20, neurons=2, N=100, t_stop=6, FDR=0.3333333333333333)[1]
Fv = Fv_calc(spks)
Rtot = len(np.concatenate(spks))/(100*6)

ISIs = []
for trial in spks:
    ISIs.append(np.diff(trial))
ISIs = np.concatenate(ISIs)
ISIs = ISIs*1000
ISIs = np.concatenate((ISIs, ISIs*-1))

ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]
    
    
# %%

fig, ax = plt.subplots(3,3)

for i in range(3):
    plot_ISIs(ISIs_df, i, ax, 0, i)
    
for i in range(3):
    plot_ISIs(ISIs_df, i+3, ax, 1, i)
    
for i in range(3):
    plot_ISIs(ISIs_df, i+6, ax, 2, i)
    

