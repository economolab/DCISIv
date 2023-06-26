# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 16:40:32 2023

@author: jpv88
"""
from pynwb import NWBFile, TimeSeries, NWBHDF5IO

import JV_utils
from tqdm import tqdm

import numpy as np

from os import listdir
from os.path import isfile, join

# %%

mypath = r"D:\\FDR Predictions DATA\\Inagaki et al 2022"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = r"D:\\FDR Predictions DATA\\Inagaki et al 2022\\"

PSTHs = []
ISI_viol = []
FDRs = []
FDR_avg = []

for file in tqdm(onlyfiles):

    nwb_read = NWBHDF5IO(mypath + file, "r").read()
    
    if nwb_read.units is None:
        continue
    
    spike_times = nwb_read.units.columns[1].data[:]
    spike_times_index = nwb_read.units.columns[0].data[:]
    trial_idx = nwb_read.units.columns[3].data[:]
    
    n_units = len(spike_times_index)
    spike_times_index = np.insert(spike_times_index, 0, 0)
    
    units = []
    n_trials = []
    uniq_trials = []
    unit_trial_idx = []
    
    for idx in range(n_units):
        idx1 = spike_times_index[idx]
        idx2 = spike_times_index[idx+1]
        
        units.append(spike_times[idx1:idx2])
        uniq_trials_temp = np.unique(trial_idx[idx1:idx2])
        unit_trial_idx.append(trial_idx[idx1:idx2])
        uniq_trials.append(uniq_trials_temp)
        n_trials.append(int(max(uniq_trials_temp) - min(uniq_trials_temp)))
        
        
        
    # length of time after go cue to use for PSTH
    T = 5
    
    ISI_viol_temp = []
    for j, unit in enumerate(units):
        viols = 0
        spikes = 0
        for trial_idx in uniq_trials[j]:
            temp = unit[unit_trial_idx[j] == trial_idx]
            temp = temp[temp <= T]
            n_spikes = len(temp)
            viols += sum(np.diff(temp) < 0.0025)
            spikes += n_spikes
            
        ISI_viol_temp.append(viols/spikes)
        
        
    bin_size = 50
    PSTHs_temp = []
    for j, unit in enumerate(units):
        PSTHs_temp.append(JV_utils.gen_PSTH(unit, n_trials[j], T, bin_size))
        
    pred = JV_utils.pred_FDR(np.stack(PSTHs_temp), ISI_viol_temp)
    
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    

# %% one file has only one neuron, and a nan is returned there, just ignore it

mean_FDR = np.nanmean(FDRs)
median_FDR = np.nanmedian(FDRs)

SEM = np.nanstd(FDRs)/(len(FDRs)-1)**(1/2)
SEmedian = ((np.pi/2)**(1/2))*SEM

