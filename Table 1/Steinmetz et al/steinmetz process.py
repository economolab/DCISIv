# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 16:59:35 2022

@author: jpv88
"""

import JV_utils
import os

import numpy as np

from tqdm import tqdm

data_dir = r'E:\\FDR Predictions DATA\\Steinmetz et al\\spikeAndBehavioralData\\allData'

exps = [x[0] for x in os.walk(data_dir)]
exps.pop(0)

clusters_str = 'spikes.clusters.npy'
times_str = 'spikes.times.npy'
goCue_times_str = 'trials.goCue_times.npy'
phy_annotation_str = 'clusters._phy_annotation.npy'

# %%

T = 4
PSTHs_full = []
ISI_viol_full = []
ISIs = []

for k in range(len(exps)):
    
    clusters = np.load(exps[k] + '\\' + clusters_str)
    times = np.load(exps[k] + '\\' + times_str)
    goCue_times = np.load(exps[k] + '\\' + goCue_times_str)
    phy_annotation = np.load(exps[k] + '\\' + phy_annotation_str)
    
    N = max(clusters).item() + 1
    
    spikes = [[] for _ in range(N)]
    
    for i, spike in enumerate(tqdm(times)):
        clust = clusters[i].item()
        spikes[clust].append(spike.item())
    
    
    end_t = max(times).item()
    num_bins = round(T/0.05)
    num_trials = len(goCue_times)
    
    PSTHs = np.zeros((N, num_bins))
    ISI_viol = np.zeros(N)
    
    for i, clust in enumerate(tqdm(spikes)):
        
        trials = [[] for _ in range(len(goCue_times))]
        
        for idx, j in enumerate(goCue_times):
            trials_spikes = np.array(clust)[(np.array(clust) < j+(T/2)) & (np.array(clust) > j-(T/2))]
            trials[idx] = np.array(trials_spikes) - j + (T/2)
            
        viols = 0
        n_spikes = 0
        for trial in trials:
            n_spikes += len(trial)
            viols += sum(np.diff(trial) < 0.0025)
            ISIs.append(np.diff(trial))
        
        ISI_viol[i] = viols/n_spikes if n_spikes != 0 else 0
            
        trials_flat = np.concatenate(trials, axis=0)
        PSTH = JV_utils.gen_PSTH(trials_flat, num_trials, T, 50)
        PSTHs[i,:] = PSTH
    
    PSTHs_full.append(PSTHs)
    ISI_viol_full.append(ISI_viol)
    
# np.save(exps[2] + '/' + 'PSTHs.npy', PSTHs)

# %%

PSTHs_flat = np.concatenate(PSTHs_full)
ISI_viol_flat = np.concatenate(ISI_viol_full)
np.save('steinmetz_PSTHs.npy', PSTHs_flat)
np.save('steinmetz_ISI_viol.npy', ISI_viol_flat)

# %%

test = np.load('steinmetz_PSTHs.npy')
test2 = np.load('steinmetz_ISI_viol.npy')


