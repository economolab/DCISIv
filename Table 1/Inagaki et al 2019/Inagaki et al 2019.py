# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 13:44:06 2023

@author: jpv88
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

import JV_utils

from tqdm import tqdm

Ts = [6, 4, 4]
PSTH_names = ['Inagaki2019_PSTHs.mat', 'Inagaki2019_PSTHs_withoutperturb.mat', 'Inagaki2019_PSTHs_withperturb.mat']
ISI_viol_names = ['Inagaki2019_ISI_viol.mat', 'Inagaki2019_ISI_viol_withoutperturb.mat', 'Inagaki2019_ISI_viol_withperturb.mat']
FDRs = []
FDR_avg = []
PSTHs = []
ISI_viol = []


path = r'C:\\Users\\jpv88\\Documents\\GitHub\\SpikeSim\\Table 1\\Inagaki et al 2019\\'
for k in range(3):
    
    mat_contents = sio.loadmat(path + PSTH_names[k])
    PSTHs_temp = mat_contents['PSTHs']
    mat_contents = sio.loadmat(path + ISI_viol_names[k])
    ISI_viol_temp = mat_contents['ISI_viol']
    
    # censor period of 1 ms assumed from looking at ISIs of data
    pred = JV_utils.pred_FDR(PSTHs_temp, ISI_viol_temp, tau_c=1)
        
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %% homogeneous testing
    
Ts = [6, 4, 4]
PSTH_names = ['Inagaki2019_PSTHs.mat', 'Inagaki2019_PSTHs_withoutperturb.mat', 'Inagaki2019_PSTHs_withperturb.mat']
ISI_viol_names = ['Inagaki2019_ISI_viol.mat', 'Inagaki2019_ISI_viol_withoutperturb.mat', 'Inagaki2019_ISI_viol_withperturb.mat']
FDRs = []
FDR_avg = []
PSTHs = []
ISI_viol = []


path = r'C:\\Users\\jpv88\\Documents\\GitHub\\SpikeSim\\Table 1\\Inagaki et al 2019\\'
for k in range(3):
    
    mat_contents = sio.loadmat(path + PSTH_names[k])
    PSTHs_temp = mat_contents['PSTHs']
    mat_contents = sio.loadmat(path + ISI_viol_names[k])
    ISI_viol_temp = mat_contents['ISI_viol']
    
    FRs = np.mean(PSTHs_temp, axis=1)
    FDRs_1 = []
    FDRs_inf = []
    
    for i, FR in enumerate(FRs):
        
        PSTH = [FR] * 100
        PSTH = np.array(PSTH)
        PSTH = PSTH.flatten()
        
        Rout_unit = [1] * 100
        Rout_unit = Rout_unit/np.linalg.norm(Rout_unit)
        Rout_unit = np.array(Rout_unit)
        
        FDRs_1.append(JV_utils.FDR_master(ISI_viol_temp[i], PSTH, Rout_unit, 1, tau=2.5, tau_c=1))
        FDRs_inf.append(JV_utils.FDR_master(ISI_viol_temp[i], PSTH, Rout_unit, float('inf'), tau=2.5, tau_c=1))
            
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