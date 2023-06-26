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

mypath = r"D:\\FDR Predictions DATA\\Xu et al"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = r"D:\FDR Predictions DATA\Xu et al\\"

PSTHs = []
ISI_viol = []
FDRs = []
FDR_avg = []

for file in tqdm(onlyfiles):

    nwb_read = NWBHDF5IO(mypath + file, "r").read()
    
    if nwb_read.units is None:
        continue
    
    if nwb_read.units.spike_times is None:
        continue
    
    spike_times = nwb_read.units.spike_times.data[:]
    spike_times_index = nwb_read.units.spike_times_index.data[:]
    cue_times = nwb_read.trials['cue_times'].data[:]
    
    n_units = len(spike_times_index)
    spike_times_index = np.insert(spike_times_index, 0, 0)
    
    units = []
    
    for idx in range(n_units):
        idx1 = spike_times_index[idx]
        idx2 = spike_times_index[idx+1]
        
        units.append(spike_times[idx1:idx2])
        
    # length of time after go cue to use for PSTH
    T = 4
    
    ISI_viol_temp = []
    aligned_units = []
    for unit in units:
        aligned_spikes = []
        viols = 0
        spikes = 0
        for cue in cue_times:
            temp = unit - cue
            temp = temp[(temp <= T) & (temp >= 0)]
            n_spikes = len(temp)
            viols += sum(np.diff(temp) < 0.0025)
            spikes += n_spikes
            aligned_spikes.extend(temp)
            
        aligned_units.append(aligned_spikes)
        ISI_viol_temp.append(viols/spikes)
    
    obs_intervals_index = nwb_read.units.columns[2].data[:]
    obs_intervals_index = np.insert(obs_intervals_index, 0, 0)
    obs_intervals = nwb_read.units.columns[3].data[:]
    n_trials = []
    for idx in range(n_units):
        idx1 = obs_intervals_index[idx]
        idx2 = obs_intervals_index[idx+1]
        
        intervals = obs_intervals[idx1:idx2]
        trials_temp = 0
        for cue in cue_times:
            for j in range(len(intervals)):
                if (cue >= intervals[j,0]) & (cue <= intervals[j,1]):
                    trials_temp += 1
                    break
        
        n_trials.append(trials_temp)
        
    bin_size = 50
    PSTHs_temp = []
    for j, unit in enumerate(aligned_units):
        PSTHs_temp.append(JV_utils.gen_PSTH(unit, n_trials[j], T, bin_size))
        
    pred = JV_utils.pred_FDR(np.stack(PSTHs_temp), ISI_viol_temp)
    
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %%

mean_FDR = np.mean(FDRs)
median_FDR = np.median(FDRs)

SEM = np.std(FDRs)/(len(FDRs)**(1/2))
SEmedian = ((np.pi/2)**(1/2))*SEM
