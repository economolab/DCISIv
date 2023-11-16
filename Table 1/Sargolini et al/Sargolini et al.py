# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:34:25 2023

@author: jpv88
"""

from pynwb import NWBFile, TimeSeries, NWBHDF5IO

import JV_utils
from tqdm import tqdm

import numpy as np

from os import listdir
from os.path import isfile, join

# %%

mypath = r"E:\\FDR Predictions DATA\\Sargolini et al"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
mypath = r"E:\\FDR Predictions DATA\\Sargolini et al\\"

FRs_full = []
ISI_viol = []
FDRs = []
FDR_avg = []

for file in tqdm(onlyfiles):
    
    FRs = []
    ISI_viol_temp = []

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
        
    for unit in units:
        rec_t = np.max(unit) - np.min(unit)
        FRs.append(len(unit)/rec_t)
        ISI_viol_temp.append(sum(np.diff(unit) < 0.0025)/len(unit))
        
    FDRs_1 = []
    FDRs_inf = []
    
    # censor period = 0.85 ms
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

    pred = (FDRs_1 + FDRs_inf)/2
    
    FDRs.extend(pred)
    FRs_full.extend(FRs)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %%

mean_FDR = np.mean(FDRs)
median_FDR = np.median(FDRs)

SEM = np.std(FDRs)/(len(FDRs)**(1/2))
SEmedian = ((np.pi/2)**(1/2))*SEM