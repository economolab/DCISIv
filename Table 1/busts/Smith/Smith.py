# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 19:05:52 2023

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

path = r'C:\\Users\\jpv88\\Documents\\GitHub\\SpikeSim\\Table 1\\Smith\\'

mat_contents = sio.loadmat(path + 'PSTHs_Pe.mat')
PSTHs_temp = mat_contents['PSTHs']
mat_contents = sio.loadmat(path + 'ISI_viol_Pe.mat')
ISI_viol_temp = mat_contents['ISI_viol']

# censor period of 0.25 ms assumed from looking at ISIs of data
pred = JV_utils.pred_FDR(PSTHs_temp, ISI_viol_temp, tau=2, tau_c=1.7)
    
FDRs.extend(pred)
PSTHs.extend(PSTHs_temp)
ISI_viol.extend(ISI_viol_temp)
FDR_avg.append(np.mean(pred))

mat_contents = sio.loadmat(path + 'PSTHs_Wi.mat')
PSTHs_temp = mat_contents['PSTHs']
mat_contents = sio.loadmat(path + 'ISI_viol_Wi.mat')
ISI_viol_temp = mat_contents['ISI_viol']

# censor period of 0.25 ms assumed from looking at ISIs of data
pred = JV_utils.pred_FDR(PSTHs_temp, ISI_viol_temp, tau=2, tau_c=1.7)
    
FDRs.extend(pred)
PSTHs.extend(PSTHs_temp)
ISI_viol.extend(ISI_viol_temp)
FDR_avg.append(np.mean(pred))


# %%

mean_FDR = np.mean(FDRs)
median_FDR = np.median(FDRs)

SEM = np.std(FDRs)/(len(FDRs)**(1/2))
SEmedian = ((np.pi/2)**(1/2))*SEM