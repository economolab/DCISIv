# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 12:11:37 2022

@author: jpv88

"""
import math
import numpy as np
import pickle
import os
import scipy

from scipy.stats import t
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


from datetime import datetime


# little n is number of trials, big N is number of bins
def spikes_to_firing_rates(spikes, n, T=6, N=100):
    delta = T/N
    bins = np.zeros(N)
    for i in range(N):
        for j in spikes:
            if (j >= i*delta) and (j < (i+1)*delta):
                bins[i] += 1
                
    bins = bins/(delta*n)
    # bins = savgol_filter(bins, 11, 4)
    return bins

# n is number of trials, T is trial length, bin_size input in milliseconds
def gen_PSTH(spikes, n, T, bin_size):
    
    spikes = np.array(spikes)
    spikes = spikes[spikes < T]
    
    bin_size = bin_size*(10**-3)
    n_bins = math.ceil(T/bin_size)
    
    bin_edges = [0]
    x = 0
    while x < n_bins:
        bin_edges.append(x*bin_size + bin_size)
        x += 1
    
    inds = np.digitize(spikes, bin_edges)
    
    bins = np.zeros(n_bins)
    
    for idx in inds:
        bins[idx-1] += 1
    
    delta = T/n_bins
    bins = bins/(delta*n)
    
    return bins

def norm_zero_to_one(data):
    mindata = min(data)
    maxdata = max(data)
    normed = [(el-mindata)/(maxdata-mindata) for el in data]
    return normed
    
def norm_neg_one_to_one(data):
    mindata = min(data)
    maxdata = max(data)
    normed = [(2*(el-mindata)/(maxdata-mindata)) - 1 for el in data]
    return normed


def point_dist_to_line(x, y, a, b, c):
    num = abs(a*x + b*y + c)
    den = (a**2 + b**2)**(1/2)
    return num/den

def lin_reg(x, y):

    x = np.array(x)
    x = x.reshape(-1, 1)
        
    y = np.array(y)
    y = y.reshape(-1, 1)
    reg = LinearRegression().fit(x, y)
    y_pred = reg.coef_*x + reg.intercept_
    print('R^2 = ' + str(reg.score(x, y)))
    R2 = reg.score(x, y)
    return y_pred, reg, R2

def scale_zero_to_x(data, x=1):
    min_data = min(data)
    max_data = max(data)
    range_data = max_data - min_data
    return([((el - min_data)/range_data)*x for el in data])

# x is length of the x axis
def norm_area(data, area, x):
    dx = x/len(data)
    data_area = np.trapz(data, dx=dx)
    return([el/(data_area/area) for el in data])

def sort_list_by_list(idx_list, sort_list):
    
    if len(idx_list) != len(sort_list):
        raise Exception('Lists must be of equal length')
        
    zipped_lists = zip(idx_list, sort_list)
    sorted_zipped_lists = sorted(zipped_lists)
    sorted_list = [element for _, element in sorted_zipped_lists]
    
    return sorted_list

def pickle_load_file(file):
    with open(file, 'rb') as pickle_file:
        var_name = file[:-7]
        globals()[var_name] = pickle.load(pickle_file)
   
def pickle_load_dir(directory=os.getcwd()):
    
    files = os.listdir(directory)
    
    for file in files:
        if file[-7:] == '.pickle':
            pickle_load_file(file)

@np.vectorize
def lin_eq(m, b, x):
    return m*x + b

def ci_95(sample):
    N = len(sample)
    df = N - 1
    std = np.std(sample)
    mean = np.mean(sample)
    low = t.ppf(0.025, df, loc=mean, scale=std)
    high = t.ppf(0.975, df, loc=mean, scale=std)
    bounds = [low, high]
    
    return bounds

def smooth_max(x, y):
    # first, make a function to linearly interpolate the data
    f = scipy.interpolate.interp1d(x, y)
    
    # resample with 1000 samples
    xx = np.linspace(x[0], x[-1], 1000)
    
    # compute the function on this finer interval
    yy = f(xx)
    
    # make a gaussian window
    window = scipy.signal.gaussian(200, 60)
    
    # convolve the arrays
    smoothed = scipy.signal.convolve(yy, window/window.sum(), mode='same')
    
    # get the maximum
    return xx[np.argmax(smoothed)]

# the final equation
def FDR_master(ISIviol, Rtot, Rout_unit, N, tau=2.5, tau_c=0):
    
    Rtot_mag = np.linalg.norm(Rtot)
    Rtot_avg = np.average(Rtot)
    Rtot_unit = Rtot/Rtot_mag
    
    tau = tau*(10**-3)
    tau_c = tau_c*(10**-3)
    n = len(Rtot)
    D = np.dot(Rtot_unit, Rout_unit)

    if N == float('inf'):
        N_correction1 = 1
        N_correction2 = 1
    else:
        N_correction1 = N/(N + 1)
        N_correction2 = (N + 1)/N
    
    sqrt1 = (Rtot_mag**2)*(D**2)
    sqrt2 = (N_correction2*Rtot_avg*ISIviol*n)/(tau - tau_c)
    Rout_mag = N_correction1*(Rtot_mag*D - (sqrt1 - sqrt2)**(0.5))
    
    Rout_avg = np.average(Rout_mag*np.array(Rout_unit))
    FDR = Rout_avg/Rtot_avg
    
    return FDR

# takes in n x m array of PSTHs, where n is the number of neurons and m is the
# number of time points, and ISI_viol, a 1D array or list of length n of ISI
# violation fractions (violations/spikes)
def pred_FDR(PSTHs, ISI_viol, tau_c=0):
    N = len(PSTHs)
    
    PSTHs_unit = np.zeros(np.shape(PSTHs))
    
    for i in range(N):
        PSTHs_unit[i,:] = PSTHs[i,:]/np.linalg.norm(PSTHs[i,:])
       
    one_FDRs = []
    inf_FDRs = []
    for i in tqdm(range(N)):
        
        other_idx = list(range(N))
        del other_idx[i]
        others_PSTH = PSTHs[other_idx,:]
        others_mean = np.mean(others_PSTH, axis=0)
        others_unit = others_mean/np.linalg.norm(others_mean)
        
        inf_FDR = FDR_master(ISI_viol[i], PSTHs[i,:], others_unit, N=float('inf'), tau_c=tau_c)
        if np.isnan(inf_FDR):
            inf_FDRs.append(1)
        else:
            inf_FDRs.append(inf_FDR)
        
        unit_FDRs = []
        for j in other_idx:
            temp = FDR_master(ISI_viol[i], PSTHs[i,:], PSTHs_unit[j,:], N=1, tau_c=tau_c)
            if np.isnan(temp):
                unit_FDRs.append(0.5)
            else:
                unit_FDRs.append(temp)
        
        one_FDRs.append(np.mean(unit_FDRs))
    
    FDRs = [np.mean([el1, el2]) for el1, el2 in zip(one_FDRs, inf_FDRs)]
    
    return FDRs

def save_sim(sim, fn, writeover=False):
    
    if writeover == False:
        now = datetime.now() # current date and time
        date_time = now.strftime("%m-%d-%Y")
        
        fn = fn + '_' + date_time
    
    # Open a file and use dump()
    with open(fn, 'wb') as file:
      
        # A new file will be created
        pickle.dump(sim, file)
        
        
def load_sim(fn):
    
    # Open a file and use dump()
    with open(fn, 'rb') as file:
      
        # A new file will be created
        data = pickle.load(file)
    
    return data
    
    
    
    




    
    