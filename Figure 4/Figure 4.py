# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:03:14 2023

@author: jpv88
"""
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
sys.path.append("..") # Adds higher directory to python modules path.

import JV_utils
import neuronsim

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from random import choices, sample
from scipy.optimize import minimize_scalar
from scipy.stats import cauchy
from scipy.interpolate import splev, splrep
from sklearn.metrics import r2_score

# %% classes

class PopMetSim:
    
    # for all matrices, row is loc, column is scale
    def __init__(self, FDR_avg, FDR_avg_old, FDR_dist_true, FDR_median, 
                 FDR_true_median, scales, locs, N_con, Rtots):
       
        self.FDR_avg = FDR_avg # predicted average FDR deleting NaNs and deleting an equal number of bottom end FDRs
        self.FDR_avg_old = FDR_avg_old # predicted average FDR ignoring NaNs
        self.FDR_dist_true = FDR_dist_true # actual average FDR
        self.FDR_median = FDR_median # predicted median FDR
        self.FDR_true_median = FDR_true_median # actual median FDR
        self.scales = scales # scale of Cauchy distribution
        self.locs = locs # loc of Cauchy distribution
        self.N_con = N_con # N_con selected from randomly
        self.Rtots = Rtots # Rtot selected from randomly

class RecTimeSim():
    
    def __init__(self, t_recs, true_FDRs, Rtots):
       
        self.t_recs = t_recs
        self.true_FDRs = true_FDRs
        self.Rtots = Rtots
        
# %%

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']
# index 512 gets kind of buggy when you try to scale it
PSTHs = np.delete(PSTHs, 512, 0)

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

# %% testing conservation of population level metrics 

k = 1000

scales = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3]
locs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.3]
FDR_avg = np.zeros((len(locs), len(scales)))
FDR_avg_old = np.zeros((len(locs), len(scales)))
FDR_dist_true = np.zeros((len(locs), len(scales)))
FDR_median = np.zeros((len(locs), len(scales)))
FDR_true_median = np.zeros((len(locs), len(scales)))


N_con = [1, 2, 5, 10]
N_con = np.array(choices(N_con, k=k), dtype='float')
N_con[N_con == 10] = float('inf')

Rtots = [4, 8, 12, 16]
Rtots = choices(Rtots, k=k)

for m, loc in enumerate(locs):
    for j, scale in enumerate(scales):
        FDRs = cauchy.rvs(loc=loc, scale=scale, size=10000)
        FDRs = FDRs[FDRs >= 0]
        FDRs = FDRs[FDRs < 1]
        FDR_dist = FDRs
        FDRs = np.random.choice(FDR_dist, size=k)
            
        PSTH_idx = list(range(len(PSTHs)))
        idx_pairs = []
        for _ in range(k):
            idx_pairs.append(sample(PSTH_idx, 2))
        
        pred_FDR = []  
        OF = []
        Fvs = []
        PSTHs_run = []
        for i in range(k):
            
            Rin = PSTHs[idx_pairs[i][0]]
            Rout = PSTHs[idx_pairs[i][1]]
        
            
            Rout_target = FDRs[i]*Rtots[i]
            Rin_target = Rtots[i] - Rout_target
            
            scale = minimize_scalar(Rout_scale_ob, args=[Rin, Rin_target], 
                                method='bounded', bounds=[0, 100]).x
            Rin = Rin*scale
        
            scale = minimize_scalar(Rout_scale_ob, 
                                args=[Rout, (FDRs[i]/(1-FDRs[i]))*np.average(Rin)],
                                method='bounded', bounds=[0, 100]).x
            
            Rout = scale*Rout
            
            Rin[Rin<0] = 0
            Rout[Rout<0] = 0
            
            Rtot = Rin + Rout
            
            center = np.average(Rin)*np.average(Rout[0])
            OF.append((np.dot(Rin, Rout)/1000 - center)/center)
            
            Fv = neuronsim.sim_Fv_PSTH4(Rin, Rout, out_refrac=2.5, 
                                        neurons=N_con[i], N=100)
            Fvs.append(Fv)
            
            PSTHs_run.append(Rtot)
        
            
        PSTHs_run = np.stack(PSTHs_run)
        pred_FDR = JV_utils.pred_FDR(PSTHs_run, Fvs)
        pred_FDR = np.array(pred_FDR)
        pred_FDR_old = pred_FDR
        
        FDR_avg_old[m,j] = np.mean(pred_FDR)
        FDR_median[m,j] = np.median(pred_FDR)
        FDR_true_median[m,j] = np.median(FDR_dist)
        
        nan_mask = (pred_FDR == 0.75)
        len_mask = sum(nan_mask)
        pred_FDR = np.delete(pred_FDR, nan_mask)
        for _ in range(len_mask): 
            pred_FDR = np.delete(pred_FDR, pred_FDR.argmin())
        
        FDR_avg[m,j] = np.mean(pred_FDR)
        FDR_dist_true[m,j] = np.mean(FDR_dist)
        
sim = PopMetSim(FDR_avg, FDR_avg_old, FDR_dist_true, FDR_median, 
                FDR_true_median, scales, locs, N_con, Rtots)

JV_utils.save_sim(sim, 'PopMetSim')

# %% load old sim data

data = JV_utils.load_sim('PopMetSim_06-19-2023')
    
        
# %% plotting median population correspondence

fig, ax = plt.subplots()
plt.scatter(data.FDR_true_median, data.FDR_median, color='b', s=20)


y_pred, reg, R2 = JV_utils.lin_reg(data.FDR_true_median, data.FDR_median)


x = [0, 0.5]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*0.5 + reg.intercept_
y = [y1.item(), y2.item()]
plt.plot(x, y, c='k', lw=2)

plt.xlim(0, 0.4)
plt.ylim(0, 0.4)
plt.plot([0, 0.4], [0, 0.4], 'k', ls='--')
plt.xlabel('True Median FDR')
plt.ylabel('Predicted Median FDR')
# plt.title('MEDIAN', fontsize=18)


plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

r2_score(data.FDR_true_median, data.FDR_median)
plt.annotate('R^2 = 0.85', (0.2, 0.05), fontsize=20)


# %% plotting mean population correspondence 

plt.scatter(data.FDR_dist_true, data.FDR_avg_old, color='b', s=20)

y_pred, reg, R2 = JV_utils.lin_reg(data.FDR_dist_true, data.FDR_avg_old)


x = [0, 0.5]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*0.5 + reg.intercept_
y = [y1.item(), y2.item()]
plt.plot(x, y, c='k', lw=2)

plt.xlim(0, 0.4)
plt.ylim(0, 0.4)
plt.plot([0, 0.4], [0, 0.4], 'k', ls='--')
plt.xlabel('True Mean FDR')
plt.ylabel('Predicted Mean FDR')

# plt.title('MEAN', fontsize=18)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

r2_score(data.FDR_dist_true, data.FDR_avg_old)
plt.annotate('R^2 = 0.87', (0.2, 0.05), fontsize=20)

# %% recording time required to get coefficient of variation 

new_save_file = 0

save_file_name = 'RecTimeSim_06-19-2023' # if using an already made save file

if new_save_file == 0:
    data = JV_utils.load_sim(save_file_name)

def CV_ob(t_rec, args):
    
    Rtot, true_FDR, num_iters = args
    
    FDRs = []
    
    Rout = Rtot*true_FDR
    Rin = Rtot - Rout
    
    Rtot_vec = np.array([Rin + Rout]*100)
    Rout_unit = [Rout]*100
    Rout_unit = np.array(Rout_unit)/np.linalg.norm(Rout_unit)
    
    for _ in range(num_iters):
        
        Fv = neuronsim.sim_Fv(Rin, Rout, t_stop=t_rec, N=1)[0]
        
        FDRs.append(JV_utils.FDR_master(Fv, Rtot_vec, Rout_unit, float('inf')))
        
    FDRs = np.array(FDRs)
    FDRs[np.isnan(FDRs)] = 1
    
    FDR_std = np.std(FDRs)
    CV = FDR_std/true_FDR
        
    return abs(CV - 0.2)

true_FDRs = [0.05, 0.3, 0.5]
num_iters = 100
Rtots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


# rows are different FDRs, columns are different Rtots
t_recs = np.zeros((len(true_FDRs), len(Rtots)))

for i, true_FDR in enumerate(true_FDRs):
    for j, Rtot in enumerate(Rtots):

        t_recs[i,j] = minimize_scalar(CV_ob, 
                                      args=[Rtot, true_FDR, num_iters], 
                                      method='bounded', 
                                      tol=10,
                                      bounds=[1, 600*60]).x
        
if new_save_file == 1:
    JV_utils.save_sim(RecTimeSim([t_recs], true_FDRs, Rtots), 'RecTimeSim')

if new_save_file == 0:
    data.t_recs.append(t_recs)
    JV_utils.save_sim(RecTimeSim(data.t_recs, true_FDRs, Rtots), save_file_name, writeover=True)
    

    
# %% plotting

data = JV_utils.load_sim(save_file_name)

t_recs = data.t_recs

t_05 = np.vstack([arr[0,:] for arr in t_recs])
t_20 = np.vstack([arr[1,:] for arr in t_recs])
t_50 = np.vstack([arr[2,:] for arr in t_recs])

