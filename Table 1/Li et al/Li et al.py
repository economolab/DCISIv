# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:03:36 2023

@author: jpv88
"""
from pynwb import NWBFile, TimeSeries, NWBHDF5IO

import JV_utils
from tqdm import tqdm

import numpy as np

from os import listdir
from os.path import isfile, join

# %%

mypath = r"E:\\FDR Predictions DATA\\Li et al\\000011"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = r"E:\\FDR Predictions DATA\\Li et al\\000011\\"

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
    cue_times = nwb_read.trials.start_time.data[:]
    
    n_units = len(spike_times_index)
    spike_times_index = np.insert(spike_times_index, 0, 0)
    
    units = []
    
    for idx in range(n_units):
        idx1 = spike_times_index[idx]
        idx2 = spike_times_index[idx+1]
        
        units.append(spike_times[idx1:idx2])
        
    # length of time after to use for PSTH centered on each go cue
    T = 6
    
    ISI_viol_temp = []
    aligned_units = []
    for unit in units:
        aligned_spikes = []
        viols = 0
        spikes = 0
        for cue in cue_times:
            temp = unit - cue
            temp = temp[(temp <= T/2) & (temp >= -T/2)]
            temp = temp + (T/2)
            n_spikes = len(temp)
            viols += sum(np.diff(temp) < 0.0025)
            spikes += n_spikes
            aligned_spikes.extend(temp)
            
        aligned_units.append(aligned_spikes)
        if spikes != 0:
            ISI_viol_temp.append(viols/spikes)
        else:
            ISI_viol_temp.append(0)
    
    obs_intervals_index = nwb_read.units.columns[7].data[:]
    obs_intervals_index = np.insert(obs_intervals_index, 0, 0)
    obs_intervals = nwb_read.units.columns[8].data[:]
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
        
    pred = JV_utils.pred_FDR(np.stack(PSTHs_temp), ISI_viol_temp, tau_c=0.85)
    
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %% testing homogeneous 

mypath = r"E:\\FDR Predictions DATA\\Li et al\\000011"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = r"E:\\FDR Predictions DATA\\Li et al\\000011\\"

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
    
    n_units = len(spike_times_index)
    spike_times_index = np.insert(spike_times_index, 0, 0)
    
    units = []
    
    for idx in range(n_units):
        idx1 = spike_times_index[idx]
        idx2 = spike_times_index[idx+1]
        
        units.append(spike_times[idx1:idx2])
        
    start = np.min(np.concatenate(units))
    end = np.max(np.concatenate(units))
        
    # length of time after to use for PSTH centered on each go cue
    T = np.ceil(end-start)
    
    ISI_viol_temp = []
    for j, unit in enumerate(units):
        
        unit = unit-start
        viols = sum(np.diff(unit) < 0.0025)
        spikes = len(unit)
        rec_time = np.max(unit) - np.min(unit)
        fr = len(unit)/rec_time
        unit = fr
        units[j] = unit

        if spikes != 0:
            ISI_viol_temp.append(viols/spikes)
        else:
            ISI_viol_temp.append(0)
        
    FRs = units
    FDRs_1 = []
    FDRs_inf = []
    for i, FR in enumerate(FRs):
        
        PSTH = [FR] * 100
        PSTH = np.array(PSTH)
        PSTH = PSTH.flatten()
        
        Rout_unit = [1] * 100
        Rout_unit = Rout_unit/np.linalg.norm(Rout_unit)
        Rout_unit = np.array(Rout_unit)
        
        FDRs_1.append(JV_utils.FDR_master(ISI_viol_temp[i], PSTH, Rout_unit, 1, tau=2.5, tau_c=0.85))
        FDRs_inf.append(JV_utils.FDR_master(ISI_viol_temp[i], PSTH, Rout_unit, float('inf'), tau=2.5, tau_c=0.85))
        
    FDRs_1 = np.array(FDRs_1)
    FDRs_1[np.isnan(FDRs_1)] = 0.5
    FDRs_inf = np.array(FDRs_inf)
    FDRs_inf[np.isnan(FDRs_inf)] = 1

    FDRs.extend((FDRs_1 + FDRs_inf)/2)
    
    
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %%

mean_FDR = np.mean(FDRs)
median_FDR = np.median(FDRs)

SEM = np.std(FDRs)/(len(FDRs)**(1/2))
SEmedian = ((np.pi/2)**(1/2))*SEM