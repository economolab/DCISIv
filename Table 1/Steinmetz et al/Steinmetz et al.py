# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 12:51:56 2023

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

path = r'C:\\Users\\jpv88\\Documents\\GitHub\\SpikeSim\\Table 1\\Steinmetz et al\\'

PSTHs_temp = np.load(path + 'steinmetz_PSTHs.npy')
ISI_viol_temp = np.load(path + 'steinmetz_ISI_viol.npy')

sessions = [1085, 1146, 1089, 2073, 1569, 1435, 1029, 1707, 1364, 1591, 1409, 1219,
            1262, 1228, 1134, 977, 727, 1350, 806, 1144, 795, 1109, 1455, 1426, 1266,
            1161, 1251, 860, 1147, 1461, 1251, 778, 854, 928, 1425, 1409, 1907,
            1041, 1793]

idx = 0

# seems to be no censor period
for j, session in enumerate(sessions):

    pred = JV_utils.pred_FDR(PSTHs_temp[idx:idx+sessions[j],:], 
                             ISI_viol_temp[idx:idx+sessions[j]])
        
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
    idx += sessions[j]
    
# %% censor period = 0, newer version

FDRs = []
FDR_avg = []
PSTHs = []
ISI_viol = []

sessions = [1085, 1146, 1089, 2073, 1569, 1435, 1029, 1707, 1364, 1591, 1409, 1219,
            1262, 1228, 1134, 977, 727, 1350, 806, 1144, 795, 1109, 1455, 1426, 1266,
            1161, 1251, 860, 1147, 1461, 1251, 778, 854, 928, 1425, 1409, 1907,
            1041, 1793]

path = r'C:\\Users\\jpv88\\Documents\\GitHub\\SpikeSim\\Table 1\\Steinmetz et al\\'

    
PSTHs_temp = np.load(path + 'PSTHs.npy')
ISI_viol_temp = np.load(path + 'ISI_viol.npy')

idx = 0

# seems to be no censor period
for j, session in enumerate(sessions):

    pred = JV_utils.pred_FDR(PSTHs_temp[idx:idx+sessions[j],:], 
                             ISI_viol_temp[idx:idx+sessions[j]], tau_c=0)
        
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
    idx += sessions[j]

    
# %% 

mean_FDR = np.mean(FDRs)
median_FDR = np.median(FDRs)

SEM = np.std(FDRs)/(len(FDRs)**(1/2))
SEmedian = ((np.pi/2)**(1/2))*SEM