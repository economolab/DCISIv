# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:48:46 2022

@author: jpv88
"""

from elephant.spike_train_generation import StationaryInverseGaussianProcess
from elephant.spike_train_generation import NonStationaryGammaProcess
from elephant.spike_train_generation import NonStationaryPoissonProcess
from elephant.spike_train_generation import StationaryPoissonProcess
from matplotlib import cm
from neo.core import AnalogSignal
from random import choice, choices, randint, sample
from scipy import stats
from tqdm import tqdm

import itertools
import matplotlib.pyplot as plt
import numpy as np
import quantities as pq
import JV_utils

SMALL_SIZE = 9
MEDIUM_SIZE = 13
BIGGER_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# %%

# rate: firing rate in Hz
# T: time length of PSTH in seconds (inhomogeneous firing)
# t_stop: time length of simulation in seconds (homogeneous firing)
# refractory_period: neuron refractory period in ms

def sim_Fv_neurons(f_t, FDR, t_stop=1000, tau=2.5, N=1):
        
    if N != float('inf'):
        N = int(neurons)
        
    if N == float('inf'):
        tau = 0
        N = 1
        
    F_v = np.zeros(N)
    Rtot = np.zeros(N)
    
    f_FP = FDR * f_t
    f_TP = f_t - f_FP
    
    f_FP = f_FP / N
    
    f_TP = pq.Quantity(f_TP, 'Hz')
    f_FP = pq.Quantity(f_FP, 'Hz')
    t_stop = pq.Quantity(t_stop, 's')
    tau = pq.Quantity(tau, 'ms')

        
    clu_TP = StationaryPoissonProcess(rate=f_TP, t_stop=t_stop, 
                                      refractory_period=tau)
    
    clu_FPs = []
    for _ in range(N):
        clu_outs.append(StationaryPoissonProcess(rate=rate_out, t_stop=t_stop,
                                                 refractory_period=refractory_period))

    
    
    spks_full = []
    desc = "R_in = " + str(rate_in) + ", R_out = " + str(rate_out)
    for i in tqdm(range(N), desc=desc):
        
        spks_in = clu_in.generate_spiketrain(as_array=True)
        spks_out = []
        for j in range(neurons):
            spks_out.append(clu_outs[j].generate_spiketrain(as_array=True))
        
        spks_out = np.concatenate(spks_out)
        
        spks_tot = np.concatenate((spks_in, spks_out))
        spks_tot = np.sort(spks_tot)
        spks_full.append(spks_tot)
        
        ISIs = np.diff(spks_tot)
        Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
        F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
        Rtot[i] = len(spks_tot)/t_stop
    
    # return np.mean(F_v)
    
    return np.mean(F_v), spks_full



# simulate multiple neurons with PSTHs being intermixed
def sim_Fv_PSTH4(PSTH_in, PSTH_out, T=6, refractory_period=2.5, N=1000, 
                 out_refrac=2.5, neurons=1):
    
    F_v = np.zeros(N)
    Rtot = np.zeros(N)
    
    if neurons != float('inf'):
        neurons = int(neurons)
        
    if neurons == float('inf'):
        out_refrac = 0
        neurons = 1
    
    refractory_period = pq.Quantity(refractory_period, 'ms')
    
    n = len(PSTH_in)
    f = n/T
    
    PSTH_out = PSTH_out/neurons
    out_refrac = pq.Quantity(out_refrac, 'ms')
    
    sig_in = AnalogSignal(PSTH_in, units='Hz', sampling_rate=f*pq.Hz)
    sig_out = AnalogSignal(PSTH_out, units='Hz', sampling_rate=f*pq.Hz)
    
    clu_in = NonStationaryPoissonProcess(rate_signal=sig_in,
                                         refractory_period=refractory_period)
    
    clu_outs = []
    for _ in range(neurons):
        clu_outs.append(NonStationaryPoissonProcess(rate_signal=sig_out,
                                              refractory_period=out_refrac))
    
    for i in tqdm(range(N)):
        
        spks_in = clu_in.generate_spiketrain(as_array=True)
        spks_out = []
        for j in range(neurons):
            spks_out.append(clu_outs[j].generate_spiketrain(as_array=True))
        
        spks_out = np.concatenate(spks_out)
        
        spks_tot = np.concatenate((spks_in, spks_out))
        spks_tot = np.sort(spks_tot)
        
        ISIs = np.diff(spks_tot)
        Nviols = sum(pq.Quantity(ISIs, 's') < refractory_period)
        F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
        Rtot[i] = len(spks_tot)/T
        
    return np.mean(F_v)



