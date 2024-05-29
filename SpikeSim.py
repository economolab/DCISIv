# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 11:48:46 2022

@author: jpv88
"""


from elephant.spike_train_generation import NonStationaryPoissonProcess
from elephant.spike_train_generation import StationaryPoissonProcess

from neo.core import AnalogSignal
from scipy.optimize import minimize_scalar

import numpy as np
import quantities as pq


# %%

# rate: firing rate in Hz
# T: time length of PSTH in seconds (inhomogeneous firing)
# t_stop: time length of simulation in seconds (homogeneous firing)
# refractory_period: neuron refractory period in ms

def sim_ISI_v_homo(f_t, FDR, t_stop=1000, tau=2.5, N=1):
        
    if N != float('inf'):
        N = int(N)
        
    if N == float('inf'):
        tau_FP = 0
        N = 1
    
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
        clu_FPs.append(StationaryPoissonProcess(rate=f_FP, t_stop=t_stop,
                                                refractory_period=tau_FP))

    spks_TP = clu_TP.generate_spiketrain(as_array=True)
    spks_FP = []
    for j in range(N):
        spks_FP.append(clu_FPs[j].generate_spiketrain(as_array=True))
    
    spks_FP = np.concatenate(spks_FP)
    
    spks_t = np.concatenate((spks_TP, spks_FP))
    spks_t = np.sort(spks_t)
    
    ISIs = np.diff(spks_t)
    Nviols = sum(pq.Quantity(ISIs, 's') < tau)
    ISI_v = Nviols/len(spks_t) if len(spks_t) != 0 else 0
    
    return ISI_v 

# simulate multiple neurons with PSTHs being intermixed
def sim_ISI_v_inhomo(f_t, FDR, f_TP, f_FP, T=6, tau=2.5, N=1, t_stop=1000):
    
    if N != float('inf'):
        tau_FP = tau
        N = int(N)
        
    if N == float('inf'):
        tau_FP = 0
        N = 1
    
    tau = pq.Quantity(tau, 'ms')
    tau_FP = pq.Quantity(tau_FP, 'ms')
    
    n = len(f_TP)
    f = n/T
    
    f_FP = f_FP/N

    sig_TP = AnalogSignal(f_TP, units='Hz', sampling_rate=f*pq.Hz)
    sig_FP = AnalogSignal(f_FP, units='Hz', sampling_rate=f*pq.Hz)
    
    clu_TP = NonStationaryPoissonProcess(rate_signal=sig_TP,
                                         refractory_period=tau)
    
    clu_FPs = []
    for _ in range(N):
        clu_FPs.append(NonStationaryPoissonProcess(rate_signal=sig_FP,
                                              refractory_period=tau_FP))
    
    n_trials = round(t_stop/T)
    Nviols = 0
    n_spikes = 0
    for i in range(n_trials):
        
        spks_TP = clu_TP.generate_spiketrain(as_array=True)
        spks_FP = []
        for j in range(N):
            spks_FP.append(clu_FPs[j].generate_spiketrain(as_array=True))
        
        spks_FP = np.concatenate(spks_FP)
        
        spks_t = np.concatenate((spks_TP, spks_FP))
        spks_t = np.sort(spks_t)
        
        ISIs = np.diff(spks_t)
        Nviols += sum(pq.Quantity(ISIs, 's') < tau)
        n_spikes += len(spks_t)
    
    ISI_v = Nviols / n_spikes
        
    return ISI_v





