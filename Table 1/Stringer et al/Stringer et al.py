# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:01:02 2023

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

path = r'E:\\FDR Predictions DATA\\Stringer et al\\'


files = ['Krebs', 'Robbins', 'Waksman']

for file in tqdm(files):
    mat_contents = sio.loadmat(path + file + '.mat')
    mat_contents = mat_contents['data_struct']
    PSTHs_temp = mat_contents[0,0]['PSTHs']
    ISI_viol_temp = mat_contents[0,0]['ISI_viol']
    sess_id_temp = mat_contents[0,0]['sess_id']
    ISI_viol_temp = ISI_viol_temp.T
    
    for sess_id in sess_id_temp:
        
        mask = (sess_id_temp==sess_id)
        mask = mask.flatten()
        
        pred = JV_utils.pred_FDR(PSTHs_temp[mask], ISI_viol_temp[mask])
        
        FDRs.extend(pred)
        PSTHs.extend(PSTHs_temp)
        ISI_viol.extend(ISI_viol_temp)
        FDR_avg.append(np.mean(pred))
        
# %% homogeneous prediction, no censor period

path = r'E:\\FDR Predictions DATA\\Stringer et al\\'
files = ['FRs_homo.mat', 'ISI_viol.mat']

mat_contents = sio.loadmat(path + files[0])
FRs = mat_contents['FRs']

mat_contents = sio.loadmat(path + files[1])
ISI_viol = mat_contents['ISI_viol']


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

# %%

ISI_viol_flat = []
for unit in ISI_viol:
    ISI_viol_flat.append(unit.item())