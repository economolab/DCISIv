# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 15:10:19 2023

@author: jpv88
"""

from pynwb import NWBFile, TimeSeries, NWBHDF5IO

import JV_utils
from tqdm import tqdm

import numpy as np
import ndx_events

from os import listdir
from os.path import isfile, join

# %%

mypath = r"E:\\FDR Predictions DATA\\Finkelstein et al"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = r"E:\\FDR Predictions DATA\\Finkelstein et al\\"

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
    cue_times = nwb_read.trials['start_time'].data[:]
    
    n_units = len(spike_times_index)
    spike_times_index = np.insert(spike_times_index, 0, 0)
    
    units = []
    
    for idx in range(n_units):
        idx1 = spike_times_index[idx]
        idx2 = spike_times_index[idx+1]
        
        units.append(spike_times[idx1:idx2])
        
    # length of time after go cue to use for PSTH
    T = 8
    
    pops = []
    ISI_viol_temp = []
    aligned_units = []
    for i, unit in enumerate(units):
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
        
        if spikes != 0:
            
            ISI_viol_temp.append(viols/spikes)
            
        else:
            ISI_viol_temp.append(0)
            pops.append(i)
    
    for idx in pops:
        ISI_viol_temp.pop(i)
        units.pop(i)
        aligned_units.pop(i)
    
    n_trials = len(cue_times)
        
    bin_size = 50
    PSTHs_temp = []
    for j, unit in enumerate(aligned_units):
        PSTHs_temp.append(JV_utils.gen_PSTH(unit, n_trials, T, bin_size))
        
    single = (nwb_read.units.quality.data[:] != 'multi')
    PSTHs_temp = np.stack(PSTHs_temp)
    PSTHs_temp = PSTHs_temp[single, :]
    ISI_viol_temp = np.array(ISI_viol_temp)[single]
    
    # censor period = 0.25 ms
    pred = JV_utils.pred_FDR(PSTHs_temp, ISI_viol_temp, tau=2.5, tau_c=0.25)
    
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %%

mean_FDR = np.mean(FDRs)
median_FDR = np.median(FDRs)

SEM = np.std(FDRs)/(len(FDRs)**(1/2))
SEmedian = ((np.pi/2)**(1/2))*SEM