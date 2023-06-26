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


path = r'C:\\Users\\jpv88\\OneDrive\\Documents\\GitHub\\Economo\\spike sorting\\Real Data FDR Predictions\\Inagaki et al 2019\\'
for k in range(3):
    
    mat_contents = sio.loadmat(path + PSTH_names[k])
    PSTHs_temp = mat_contents['PSTHs']
    mat_contents = sio.loadmat(path + ISI_viol_names[k])
    ISI_viol_temp = mat_contents['ISI_viol']
    
    
    pred = JV_utils.pred_FDR(PSTHs_temp, ISI_viol_temp)
        
    FDRs.extend(pred)
    PSTHs.extend(PSTHs_temp)
    ISI_viol.extend(ISI_viol_temp)
    FDR_avg.append(np.mean(pred))
    
# %%

mean_FDR = np.mean(FDRs)
median_FDR = np.median(FDRs)

SEM = np.std(FDRs)/(len(FDRs)**(1/2))
SEmedian = ((np.pi/2)**(1/2))*SEM