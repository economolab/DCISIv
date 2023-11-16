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
from scipy import stats
from scipy import interpolate
from scipy.ndimage import gaussian_filter1d

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
        
class ISIvSim():
    
    def __init__(self, Fvs_low, Fvs_mid, Fvs_high):
        
        self.Fvs_low = Fvs_low
        self.Fvs_mid = Fvs_mid
        self.Fvs_high = Fvs_high
        
class fullSim():
    
    def __init__(self, pred_FDR, covs, FDRs, Rtots, N_con):
       
        self.pred_FDR = pred_FDR
        self.covs = covs
        self.FDRs = FDRs
        self.Rtots = Rtots
        self.N_con = N_con
        
# %% picking good vals
        
@np.vectorize
def kleinfeld_eq(Rtot, F_v, tviol=0.0025, t_c=0):

    a = -2*(tviol-t_c)*Rtot
    b = 2*(tviol-t_c)*Rtot
    c = -F_v
    
    if Rtot != 0:
        FDR = (-b + (b**2 - 4*a*c)**(1/2))/(2*a)
    else:
        FDR = 0
    
    if b**2 - 4*a*c < 0:
        FDR = float('NaN')
    
    if isinstance(FDR, complex):
        FDR = 0.5

    return FDR

Rtot = 8
print(kleinfeld_eq(Rtot, 0.005))

Rout = kleinfeld_eq(Rtot, 0.005)*Rtot
Rin = Rtot - Rout
        
        
# %% ISIv is normally distributed and variance scales inversely with recording 
#### time, panel A

# low, mid, and high recording times
Fvs_low = []
Fvs_mid = []
Fvs_high = [] 

for _ in range(1000):
    Fvs_low.append(neuronsim.sim_Fv(Rin, Rout, t_stop=600, N=1, out_refrac=2.5)[0])
    Fvs_mid.append(neuronsim.sim_Fv(Rin, Rout, t_stop=1800, N=1, out_refrac=2.5)[0])
    Fvs_high.append(neuronsim.sim_Fv(Rin, Rout, t_stop=7200, N=1, out_refrac=2.5)[0])
    
sim = ISIvSim(Fvs_low, Fvs_mid, Fvs_high)

JV_utils.save_sim(sim, 'ISIvSim')    
    
# %%

sim = JV_utils.load_sim('ISIvSim_07-13-2023')

Fvs_low = sim.Fvs_low
Fvs_mid = sim.Fvs_mid
Fvs_high = sim.Fvs_high

def fit_plot_pdf(Fvs, ax, c):
    
    mu, sigma = stats.norm.fit(Fvs)
    points = np.linspace(stats.norm.ppf(0.0001,loc=mu,scale=sigma),
                     stats.norm.ppf(0.9999,loc=mu,scale=sigma),1000)
    pdf = stats.norm.pdf(points,loc=mu,scale=sigma)
    
    ax.plot(points, pdf, color=c)
    # plt.hist(Fvs, color=c)
    plt.fill_between(points,pdf, color=c, alpha=0.5)
    


fig, ax = plt.subplots()

fit_plot_pdf(Fvs_low, ax, 'r')
fit_plot_pdf(Fvs_mid, ax, 'g')
fit_plot_pdf(Fvs_high, ax, 'b')

ax.set_ylim(0)

ax.vlines(0.005, 0, 1000)

def make_square_axes(ax):
    """Make an axes square in screen units.

    Should be called after plotting.
    """
    ax.set_aspect(1 / ax.get_data_ratio())


make_square_axes(ax)

# ax.set_aspect('equal')
# ax.set_xlim(-0.025, 1.025)
# ax.set_ylim(-0.025, 1.025)

plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

# %% load in and preprocess PSTHs

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']

t = np.linspace(0, 6, 120)
t_new = np.linspace(0, 6, 1000)

PSTHs_new = []
for i in range(len(PSTHs)):
    smoothed = gaussian_filter1d(PSTHs[i], 3)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    PSTHs_new.append(y_new)
    
PSTHs = np.vstack(PSTHs_new)

# %% predicted vs true across a range of conditions, panel B, simulation

from random import choices, sample, uniform
from scipy.optimize import minimize_scalar

from JV_utils import FDR_master

def Rout_scale_ob(scale, args):
    Rout_old, Rout_avg_new = args
    return abs(np.average(scale*Rout_old) - Rout_avg_new)

done = 0
while done == 0:
    
    k = 100
    
    N_con = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    N_con = np.array(choices(N_con, k=k), dtype='float')
    N_con[N_con == 10] = float('inf')
    
    Rtots = [1, 10]
    Rtots = np.random.uniform(Rtots[0], Rtots[1], size=k)
    
    FDRs = []
    for i in range(k):
        FDR_max = N_con[i]/(N_con[i] + 1)
        if N_con[i] == float('inf'):
            FDR_max = 1
        FDRs.append(uniform(0, FDR_max))
        
    PSTH_idx = list(range(len(PSTHs)))
    idx_pairs = []
    for _ in range(k):
        idx_pairs.append(sample(PSTH_idx, 2))
    
    pred_FDR = []  
    
    covs = []    
        
    # store the simulated PSTHs
    Rtots_matrix = np.zeros((k, np.shape(PSTHs)[1]))
    Fvs = []
    
    for i in range(k):
        
        Rin = PSTHs[idx_pairs[i][0]]
        Rout = PSTHs[idx_pairs[i][1]]
        Rin[Rin<0] = 0
        Rout[Rout<0] = 0
        
        Rout_target = FDRs[i]*Rtots[i]
        Rin_target = Rtots[i] - Rout_target
        
        scale = minimize_scalar(Rout_scale_ob, args=[Rin, Rin_target], 
                            method='bounded', bounds=[0, 100]).x
        Rin = Rin*scale
    
        scale = minimize_scalar(Rout_scale_ob, 
                            args=[Rout, (FDRs[i]/(1-FDRs[i]))*np.average(Rin)],
                            method='bounded', bounds=[0, 100]).x
        
        Rout = scale*Rout
        
        Rtot = Rin + Rout
        
        Rtots_matrix[i,:] = Rtot
        
        center = np.average(Rin)*np.average(Rout[0])
        covs.append(np.cov(Rin, Rout)[0,1])
        
        Fv = neuronsim.sim_Fv_PSTH4(Rin, Rout, out_refrac=2.5, 
                                    neurons=N_con[i], N=100)
        
        Fvs.append(Fv)
        
        single_pred = FDR_master(Fv, Rtot, Rout/np.linalg.norm(Rout), N_con[i])
        
        if np.isnan(single_pred):
            single_pred = N_con[i]/(N_con[i] + 1)
            if N_con[i] == float('inf'):
                single_pred = 1
        
        pred_FDR.append(single_pred)
        
    # if sum(np.isnan(np.array(pred_FDR))) == 0:
    done = 1
        
    sim = fullSim(pred_FDR, covs, FDRs, Rtots, N_con)
    
    JV_utils.save_sim(sim, 'fullSim')

# %% plotting

import matplotlib as mpl

sim = JV_utils.load_sim('fullSim_07-15-2023_1')
pred_FDR = sim.pred_FDR
covs = sim.covs
FDRs = sim.FDRs
Rtots = sim.Rtots
N_con = sim.N_con

pred_FDR = np.array(pred_FDR)
FDRs = np.array(FDRs)


idxs = np.array(N_con) == 1
fig, ax = plt.subplots()
plt.scatter(FDRs, pred_FDR, c='blue', s=24)
plt.plot([0, 1], [0, 1], ls='dashed', c='k', lw=2)
plt.xlabel('True FDR', fontsize=16)
plt.ylabel('Predicted FDR', fontsize=16)
plt.text(0.3, 0.1, '$R^2$ = 0.98', fontsize=16)

y_pred, reg, R2 = JV_utils.lin_reg(FDRs, pred_FDR)

x = [0, 1]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*1 + reg.intercept_
y = [y1.item(), y2.item()]
plt.plot(x, y, c='k', lw=2)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

ax.set_aspect('equal')
ax.set_xlim(-0.025, 1.025)
ax.set_ylim(-0.025, 1.025)
plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

        

# %% load in and preprocess PSTHs

mat_contents = sio.loadmat('hidehiko_PSTHs')
PSTHs = mat_contents['PSTHs']

t = np.linspace(0, 6, 120)
t_new = np.linspace(0, 6, 1000)

PSTHs_new = []
for i in range(len(PSTHs)):
    smoothed = gaussian_filter1d(PSTHs[i], 3)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    y_new = f(t_new)
    PSTHs_new.append(y_new)
    
PSTHs = np.vstack(PSTHs_new)

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
            
            
            single_pred = JV_utils.FDR_master(Fv, Rtot, Rout/np.linalg.norm(Rout), N_con[i])
            
            if np.isnan(single_pred):
                single_pred = N_con[i]/(N_con[i] + 1)
                if N_con[i] == float('inf'):
                    single_pred = 1
            
            pred_FDR.append(single_pred)
        
        pred_FDR = np.array(pred_FDR)
        pred_FDR_old = pred_FDR
        
        FDR_median[m,j] = np.median(pred_FDR)
        FDR_true_median[m,j] = np.median(FDR_dist)
        
        FDR_avg[m,j] = np.mean(pred_FDR)
        FDR_dist_true[m,j] = np.mean(FDR_dist)
        
sim = PopMetSim(FDR_avg, FDR_avg_old, FDR_dist_true, FDR_median, 
                FDR_true_median, scales, locs, N_con, Rtots)

JV_utils.save_sim(sim, 'PopMetSim')

# %% load old sim data

data = JV_utils.load_sim('PopMetSim_07-17-2023')

fig, ax = plt.subplots()
plt.scatter(data.FDR_true_median, data.FDR_median, color='b', s=20)


y_pred, reg, R2 = JV_utils.lin_reg(data.FDR_true_median, data.FDR_median)


x = [0, 0.4]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*0.4 + reg.intercept_
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

ax.set_aspect('equal')

ax.set_xlim(-0.025, 0.425)
ax.set_ylim(-0.025, 0.425)

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

r2_score(data.FDR_true_median, data.FDR_median)
plt.annotate('R^2 = 0.85', (0.2, 0.05), fontsize=20)


# %% plotting mean population correspondence 


fig, ax = plt.subplots()
plt.scatter(data.FDR_dist_true, data.FDR_avg, color='b', s=20)

y_pred, reg, R2 = JV_utils.lin_reg(data.FDR_dist_true, data.FDR_avg)


x = [0, 0.4]
y1 = reg.coef_*0 + reg.intercept_
y2 = reg.coef_*0.4 + reg.intercept_
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


ax.set_aspect('equal')

ax.set_xlim(-0.025, 0.425)
ax.set_ylim(-0.025, 0.425)

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'

r2_score(data.FDR_dist_true, data.FDR_avg)
plt.annotate('R^2 = 0.87', (0.2, 0.05), fontsize=20)

# %% recording time required to get coefficient of variation 

n = 9
for _ in range(n):
    new_save_file = 0
    
    save_file_name = 'RecTimeSim_07-21-2023' # if using an already made save file
    
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
    
    true_FDRs = [0.05, 0.1, 0.3]
    num_iters = 1000
    Rtots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    
    # rows are different FDRs, columns are different Rtots
    t_recs = np.zeros((len(true_FDRs), len(Rtots)))
    
    for i, true_FDR in enumerate(true_FDRs):
        for j, Rtot in enumerate(Rtots):
    
            t_recs[i,j] = minimize_scalar(CV_ob, 
                                          args=[Rtot, true_FDR, num_iters], 
                                          method='bounded', 
                                          tol=10,
                                          bounds=[1, 600*200]).x
            
    if new_save_file == 1:
        JV_utils.save_sim(RecTimeSim([t_recs], true_FDRs, Rtots), 'RecTimeSim')
    
    if new_save_file == 0:
        data.t_recs.append(t_recs)
        JV_utils.save_sim(RecTimeSim(data.t_recs, true_FDRs, Rtots), save_file_name, writeover=True)
    

    
# %% plotting

data = JV_utils.load_sim(save_file_name)

t_recs = data.t_recs

t_05 = np.vstack([arr[0,:] for arr in t_recs])
t_10 = np.vstack([arr[1,:] for arr in t_recs])
t_30 = np.vstack([arr[2,:] for arr in t_recs])

t_05_means = np.mean(t_05, axis=0)
t_10_means = np.mean(t_10, axis=0)
t_30_means = np.mean(t_30, axis=0)

def smooth_line(x, y):
    
    t = x
    t_new = np.linspace(1, 10, 1000)

    smoothed = gaussian_filter1d(y, 0.5)
    f = interpolate.interp1d(t, smoothed, kind='cubic')
    return f(t_new)
    
Rtots = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
t_new = np.linspace(1, 10, 1000)

# t_05_means = smooth_line(Rtots, t_05_means)
# t_10_means = smooth_line(Rtots, t_10_means)
# t_30_means = smooth_line(Rtots, t_30_means)

t_05_stds = np.std(t_05, axis=0)
t_10_stds = np.std(t_10, axis=0)
t_30_stds = np.std(t_30, axis=0)

# t_05_stds = smooth_line(Rtots, t_05_stds)
# t_10_stds = smooth_line(Rtots, t_10_stds)
# t_30_stds = smooth_line(Rtots, t_30_stds)

t_05_means = t_05_means/60
t_10_means = t_10_means/60
t_30_means = t_30_means/60
t_05_stds = t_05_stds/60
t_10_stds = t_10_stds/60
t_30_stds = t_30_stds/60

fig, ax = plt.subplots()
ax.plot(Rtots, t_05_means, c='b')
ax.plot(Rtots, t_10_means, c ='g')
ax.plot(Rtots, t_30_means, c='r')

plt.fill_between(Rtots, t_05_means-t_05_stds*2, 
                  t_05_means+t_05_stds*2, color='b', alpha=0.1)
plt.fill_between(Rtots, t_10_means-t_10_stds*2, 
                  t_10_means+t_10_stds*2, color='g', alpha=0.1)
plt.fill_between(Rtots, t_30_means-t_30_stds*2, 
                  t_30_means+t_30_stds*2, color='r', alpha=0.1)

t_05_means = np.mean(t_05, axis=0)/60
plt.scatter(Rtots, t_05_means, c='b', s=10)

t_10_means = np.mean(t_10, axis=0)/60
plt.scatter(Rtots, t_10_means, c='g', s=10)

t_30_means = np.mean(t_30, axis=0)/60
plt.scatter(Rtots, t_30_means, c='r', s=10)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.yscale('log')

plt.grid(axis='both', which='both', ls ='--', alpha=0.3, lw=1)
plt.ylabel('Recording Time (min)')
plt.xlabel('R (Hz)')
plt.xticks([1, 2, 3, 4,5, 6, 7,8,9, 10])
ax.set_aspect(1.4)

plt.tight_layout()

mpl.rcParams['image.composite_image'] = False
plt.rcParams['svg.fonttype'] = 'none'


