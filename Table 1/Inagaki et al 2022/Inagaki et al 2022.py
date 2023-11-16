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

mypath = r"E:\\FDR Predictions DATA\\Inagaki et al 2022"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = r"E:\\FDR Predictions DATA\\Inagaki et al 2022\\"

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
        
    pred = JV_utils.pred_FDR(np.stack(PSTHs_temp), ISI_viol_temp, tau_c=0.25)
    
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %% homogeneous testing

mypath = r"E:\\FDR Predictions DATA\\Inagaki et al 2022"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = r"E:\\FDR Predictions DATA\\Inagaki et al 2022\\"

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
    
    FRs = []
    ISI_viol_temp = []
    for j, unit in enumerate(units):
        viols = 0
        spikes = 0
        time = 0
        
        for trial_idx in uniq_trials[j]:
            temp = unit[unit_trial_idx[j] == trial_idx]
            temp = temp[temp <= T]
            n_spikes = len(temp)
            viols += sum(np.diff(temp) < 0.0025)
            spikes += n_spikes
            time += T
        
        FRs.append(spikes/time)
            
        ISI_viol_temp.append(viols/spikes)
    
    FDRs_1 = []
    FDRs_inf = []
    
    for i, FR in enumerate(FRs):
        
        PSTH = [FR] * 100
        PSTH = np.array(PSTH)
        PSTH = PSTH.flatten()
        
        Rout_unit = [1] * 100
        Rout_unit = Rout_unit/np.linalg.norm(Rout_unit)
        Rout_unit = np.array(Rout_unit)
        
        FDRs_1.append(JV_utils.FDR_master(ISI_viol_temp[i], PSTH, Rout_unit, 1, tau=2.5, tau_c=0.25))
        FDRs_inf.append(JV_utils.FDR_master(ISI_viol_temp[i], PSTH, Rout_unit, float('inf'), tau=2.5, tau_c=0.25))
            
    FDRs_1 = np.array(FDRs_1)
    FDRs_1[np.isnan(FDRs_1)] = 0.5
    FDRs_inf = np.array(FDRs_inf)
    FDRs_inf[np.isnan(FDRs_inf)] = 1
    
    FDRs.extend((FDRs_1 + FDRs_inf)/2)
        

    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    

# %% one file has only one neuron, and a nan is returned there, just ignore it

mean_FDR = np.nanmean(FDRs)
median_FDR = np.nanmedian(FDRs)

SEM = np.nanstd(FDRs)/(len(FDRs)-1)**(1/2)
SEmedian = ((np.pi/2)**(1/2))*SEM

