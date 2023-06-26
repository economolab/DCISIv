# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 16:42:16 2023

@author: jpv88
"""

import scipy.io as sio

import JV_utils

data_dir = 'C:\\Users\\jpv88\\OneDrive\\Documents\\GitHub\\Economo\\spike sorting\\Real Data FDR Predictions\\Mallory et al\\'

mat_fname = 'mallory_ISI_viol.mat'
ISI_viol = sio.loadmat(data_dir + mat_fname)['ISI_viol']
ISI_viol = ISI_viol.flatten()

mat_fname = 'mallory_PSTHs.mat'
PSTHs = sio.loadmat(data_dir + mat_fname)['PSTHs']

FDRs = JV_utils.pred_FDR(PSTHs, ISI_viol)