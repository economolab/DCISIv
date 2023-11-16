# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 15:24:59 2023

@author: jpv88
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import JV_utils

from tqdm import tqdm

FDRs = []
FDR_avg = []
PSTHs = []
ISI_viol = []

path = r'C:\\Users\\jpv88\\OneDrive\\Documents\\GitHub\\Economo\\spike sorting\\Real Data FDR Predictions\\Juavinett et al\\'


files = ['NP8_001', 'NP8_002', 'NP8_003', 'NP8_004', 'NP8_005', 'NP8_006',
         'NP9_000', 'NP9_001', 'NP9_002', 'NP9_003', 'NP9_005', 'NP14_000',
         'NP14_001', 'NP14_003', 'NP14_004', 'NP14_006', 'NP15_000', 
         'NP15_001', 'NP15_002', 'NP15_003', 'NP15_004', 'NP16_000', 
         'NP16_001', 'NP16_002', 'NP16_003', 'NP16_004']

for file in tqdm(files):
    mat_contents = sio.loadmat(path + file + '_spikes.mat_PSTHs.mat')
    PSTHs_temp = mat_contents['PSTHs']
    mat_contents = sio.loadmat(path + file + '_spikes.mat_ISIviol.mat')
    ISI_viol_temp = mat_contents['ISI_viol']
    ISI_viol_temp = ISI_viol_temp.T
    
    
    pred = JV_utils.pred_FDR(PSTHs_temp, ISI_viol_temp)
        
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %% homogeneous version, censor period = 0

FDRs = []
FDR_avg = []
PSTHs = []
ISI_viol = []

path = r'C:\\Users\\jpv88\\Documents\\GitHub\\SpikeSim\\Table 1\\Juavinett et al\\'

files = ['Juavinett_FRs', 'Juavinett_ISI_viol']


mat_contents = sio.loadmat(path + files[0])
FRs = mat_contents['FRs']
FRs = FRs.flatten()

mat_contents = sio.loadmat(path + files[1])
ISI_viol = mat_contents['ISI_viol']
ISI_viol = ISI_viol.flatten()

FDRs_1 = []
FDRs_inf = []
for i, FR in enumerate(FRs):
    
    PSTH = [FR] * 100
    PSTH = np.array(PSTH)
    PSTH = PSTH.flatten()
    
    Rout_unit = [1] * 100
    Rout_unit = Rout_unit/np.linalg.norm(Rout_unit)
    Rout_unit = np.array(Rout_unit)
    
    FDRs_1.append(JV_utils.FDR_master(ISI_viol[i], PSTH, Rout_unit, 1, tau=2.5, tau_c=0))
    FDRs_inf.append(JV_utils.FDR_master(ISI_viol[i], PSTH, Rout_unit, float('inf'), tau=2.5, tau_c=0))
    
FDRs_1 = np.array(FDRs_1)
FDRs_1[np.isnan(FDRs_1)] = 0.5
FDRs_inf = np.array(FDRs_inf)
FDRs_inf[np.isnan(FDRs_inf)] = 1

FDRs = (FDRs_1 + FDRs_inf)/2
    
# %%

mean_FDR = np.mean(FDRs)
median_FDR = np.median(FDRs)

SEM = np.std(FDRs)/(len(FDRs)**(1/2))
SEmedian = ((np.pi/2)**(1/2))*SEM