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
import matplotlib as mpl
import numpy as np
import pandas as pd
import scipy.io as sio

from numpy import linalg as LA
from scipy import integrate, interpolate
from scipy.ndimage import gaussian_filter1d

import JV_utils
import neuronsim

# %%

resimFR = 1
resimOverlap = 0
resimNcon = 0

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
def plot_ISIs(ISIs_df, k, ax, i, j, c):
    
    ISIs = ISIs_df['ISIs'][k]
    Fv = ISIs_df['Fv'][k]
    
    bins = np.arange(-25.5, 25.5, 0.5)
    ax[i,j].hist(ISIs, bins, facecolor=c, edgecolor='black', zorder=3, linewidth=0.5)
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

def calc_Rin(Rtot, Fv, tau):
    
    a = (2*tau)/Rtot
    b = -2*tau
    c = Fv
    
    Rin = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    
    return Rin

if resimFR == 1:
    
    done = 0
    
    while done == 0:

        ISIs_df = pd.DataFrame(columns=['ISIs', 'Fv', 'Rtot'])
        
        Rtots = [20, 6.2, 4]
        
        ISIv = 0.005
        
        Rin = [calc_Rin(Rtot, ISIv, 0.0025) for Rtot in Rtots]
        Rout = np.array(Rtots) - np.array(Rin)
        Rout = list(Rout)
        
        FDRs = np.array(Rout)/np.array(Rtots)
        
        for i in range(3):
            
            ISIs, Fv, Rtot = gen_ISIs(Rin[i], Rout[i], 100, 6, 2.5)
            ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]
        
        # if every ISI violation fraction rounded to 3 decimals is the same
        if np.all(np.round(ISIs_df['Fv'].values, decimals=3) == ISIv):
            done = 1
            
    JV_utils.save_sim(ISIs_df, 'ChangeFireRate')
    
    fig, ax = plt.subplots(3,3)

    for i in range(3):
        plot_ISIs(ISIs_df, i, ax, 0, i, 'blue')



# %% changing PSTH overlap

if resimOverlap == 1:


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
    
    rogue1 = np.roll(rogue_base, 42)
    rogue2 = np.roll(rogue_base, 103)
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
    true_scale = 200
    rogue_scale = 53
    
    Rin_int = integrate.simpson(true1*true_scale)
    Rout_int = integrate.simpson(rogue1*rogue_scale)
    
    FDR = Rout_int/(Rin_int + Rout_int)
    ISI_viols = []
    for i in range(3):
        Rviol = 2*0.0025*np.dot(true1*true_scale, rogues[i]*rogue_scale)/len(true1)
        ISI_viols.append(Rviol/np.mean(np.array(true1*true_scale) + np.array(rogues[i]*rogue_scale)))
    
# %%


    done = 0
    while done == 0:
        
        ISIs_df = pd.DataFrame(columns=['ISIs', 'Fv', 'Rtot'])
    
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
        
            ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]
            
        # if every ISI violation fraction rounded to 3 decimals is the same
        if np.round(ISIs_df['Fv'].values[0], decimals=3) == 0.005:
            print('check1')
            if np.round(ISIs_df['Fv'].values[1], decimals=3) == 0.010:
                print('check2')
                if np.round(ISIs_df['Fv'].values[2], decimals=3) == 0.025:
                    print('check3')
                    done = 1
        
    JV_utils.save_sim(ISIs_df, 'ChangeOverlap')
    
    fig, ax = plt.subplots(3,3)

    for i in range(3):
        plot_ISIs(ISIs_df, i, ax, 0, i, 'blue')
    
# %% generate PSTH traces

fig, ax = plt.subplots()
plt.plot(true1*true_scale)
plt.plot(rogue1*rogue_scale)

fig, ax = plt.subplots()
plt.plot(true1*true_scale)
plt.plot(rogue2*rogue_scale)

fig, ax = plt.subplots()
plt.plot(true1*true_scale)
plt.plot(rogue3*rogue_scale)

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
    
# %% changing confounding neuron count

if resimNcon == 1:
    
    ISI_viol = 0.01
    Rtot = 10
    Rtot_vec = [Rtot] * 100
    Rout = 1
    Rout_vec = [Rout] * 100
    Rout_mag = np.linalg.norm(Rout_vec)
    Rout_unit = Rout_vec/Rout_mag
    
    N_cons = [1, 2, float('inf')]
    FDRs = []
    for N_con in N_cons:
        FDRs.append(JV_utils.FDR_master(ISI_viol, Rtot_vec, Rout_unit, N_con))
        
    done = 0
    while done == 0:
    
        ISIs_df = pd.DataFrame(columns=['ISIs', 'Fv', 'Rtot'])
    
        ISIs, Fv, Rtot = gen_ISIs((1-FDRs[2])*Rtot, FDRs[2]*Rtot, 100, 6, 0)
        ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]
        
        ISIs, Fv, Rtot = gen_ISIs((1-FDRs[0])*Rtot, FDRs[0]*Rtot, 100, 6, 2.5)
        ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]
            
        spks = neuronsim.sim_Fv_neurons(Rtot, neurons=2, N=100, t_stop=6, FDR=FDRs[1])[1]
        Fv = Fv_calc(spks)
        Rtot = len(np.concatenate(spks))/(100*6)
        
        ISIs = []
        for trial in spks:
            ISIs.append(np.diff(trial))
        ISIs = np.concatenate(ISIs)
        ISIs = ISIs*1000
        ISIs = np.concatenate((ISIs, ISIs*-1))
        
        ISIs_df.loc[len(ISIs_df.index)] = [ISIs, Fv, Rtot]
        
        # if every ISI violation fraction rounded to 3 decimals is the same
        if np.all(np.round(ISIs_df['Fv'].values, decimals=3) == ISI_viol):
            done = 1
    
    JV_utils.save_sim(ISIs_df, 'ChangeNcon')
    
    fig, ax = plt.subplots(3,3)

    for i in range(3):
        plot_ISIs(ISIs_df, i, ax, 0, i)
    
    
# %%

fig, ax = plt.subplots(3,3)

colors = ['blue', 'green', 'red']

for i in range(3):
    ISIs_df = JV_utils.load_sim('ChangeFireRate_06-30-2023')
    plot_ISIs(ISIs_df, i, ax, 0, i, colors[i])
    
for i in range(3):
    ISIs_df = JV_utils.load_sim('ChangeOverlap_06-30-2023')
    plot_ISIs(ISIs_df, i, ax, 1, i, colors[i])
    
for i in range(3):
    ISIs_df = JV_utils.load_sim('ChangeNcon_06-29-2023')
    plot_ISIs(ISIs_df, i, ax, 2, i, colors[i])
    
mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'
    

