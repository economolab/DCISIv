# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:31:41 2024

@author: jpv88
"""

import warnings

import numpy as np
import pandas as pd

from random import sample
from tqdm import tqdm


class DCISIv():
    
    def __init__(self, f_t, ISI_v, N=(1, float('inf')), tau=2.5, tau_c=0):
        
        self.f_t = f_t
        self.ISI_v = ISI_v
        self.tau = tau 
        self.tau_c = tau_c
        
        if hasattr(N, "__len__"):
            self.N = N
        else:
            self.N = [N]
        
        self.tau = self.tau*(10**-3)
        self.tau_c = self.tau_c*(10**-3)
        self.tau_e = self.tau - self.tau_c
        
        self.max_compare = 10
        
        self.f_t = np.asarray(self.f_t)
        match self.f_t.ndim:
            case 1: 
                self.model = 'homo'
            case 2:
                self.model = 'inhomo'
            case _:
                raise Exception('Input f_t must be either 1 or 2 dimensional array')
                
        match self.model:
            case 'homo':
                if any(self.f_t <= 0):
                    raise Exception('All input f_t must be greater than 0')
                FDRs = self.pred_FDR_homo()
            case 'inhomo':
                FDRs = self.pred_FDR_inhomo()
                
        
        FDR_means = np.mean(FDRs, axis=1)
        FDRs = pd.DataFrame(FDRs)
        FDRs.columns = [str(x) for x in self.N]
        FDRs.insert(len(FDRs.columns), 'mean', FDR_means)
        
        self.res = {}
        self.res['FDRs'] = FDRs
        self.res['mean'] = np.mean(FDRs['mean'].values)
        self.res['median'] = np.median(FDRs['mean'].values)
        
        
    # the final equation
    def FDR_inhomo(self, params):
        
        f_t = params['f_t']
        ISI_v = params['ISI_v']
        N = params['N'] 
        tau_e = params['tau_e']
        f_FP_unit = params['f_FP_unit']
    
        f_t_mag = np.linalg.norm(f_t)
        f_t_unit = f_t / f_t_mag
        f_t_avg = np.average(f_t)

        n = len(f_t)
        D = np.dot(f_t_unit, f_FP_unit)

        if N == float('inf'):
            N_correction1 = 1
            N_correction2 = 1
        else:
            N_correction1 = N/(N + 1)
            N_correction2 = (N + 1)/N
        
        sqrt1 = (f_t_mag**2)*(D**2)
        sqrt2 = (N_correction2*f_t_avg*ISI_v*n)/(tau_e)
        f_FP_mag = N_correction1*(f_t_mag*D - (sqrt1 - sqrt2)**(0.5))
        
        f_FP_avg = np.average(f_FP_mag*np.array(f_FP_unit))
        FDR = f_FP_avg/f_t_avg
        
        if np.isnan(FDR):
            if N == float('inf'):
                FDR = 1
            else:
                FDR = N / (N + 1)
        
        return FDR
    
    # the final equation
    def FDR_homo(self, params):
        
        f_t = params['f_t']
        ISI_v = params['ISI_v']
        N = params['N'] 
        tau_e = params['tau_e']

        if N == float('inf'):
            N_correction1 = 1
            N_correction2 = 1
        else:
            N_correction1 = N/(N + 1)
            N_correction2 = (N + 1)/N
        
        sqrt1 = 1
        sqrt2 = (N_correction2*ISI_v)/(f_t*tau_e)
        FDR = N_correction1*(1 - (sqrt1 - sqrt2)**(0.5))
        
        if np.isnan(FDR):
            if N == float('inf'):
                FDR = 1
            else:
                FDR = N / (N + 1)
        
        return FDR
        
    def build_params_inhomo(self, i, N, f_FP_unit):
        
        params = {}
        params['f_t'] = self.f_t[i,:]
        params['ISI_v'] = self.ISI_v[i]
        params['N'] = N
        params['tau_e'] = self.tau_e
        params['f_FP_unit'] = f_FP_unit
        
        return params
    
    def build_params_homo(self, i, N):
        
        params = {}
        params['f_t'] = self.f_t[i]
        params['ISI_v'] = self.ISI_v[i]
        params['N'] = N
        params['tau_e'] = self.tau_e
        
        return params
        
    # takes in n x m array of PSTHs, where n is the number of neurons and m is the
    # number of time points, and ISI_viol, a 1D array or list of length n of ISI
    # violation fractions (violations/spikes)
    def pred_FDR_inhomo(self):
        
        n_neurons = self.f_t.shape[0]
        n_N = len(self.N)
        FDRs = np.zeros((n_neurons, n_N))
        
        f_t_unit = np.zeros(self.f_t.shape)
        for i in range(n_neurons):
            f_t_unit[i,:] = self.f_t[i,:] / np.linalg.norm(self.f_t[i,:])
            
        # i = neuron index
        # j = N index 
        
        pbar = tqdm(total=n_neurons * n_N, 
                    desc='Calculating...')
        
        for j in range(n_N):
            
            N = self.N[j]
            
            for i in range(n_neurons):
                
                other_idx = list(range(n_neurons))
                del other_idx[i]
                other_f_t_unit = f_t_unit[other_idx,:]
                
                if N == float('inf'):
                    f_FP_unit = np.sum(other_f_t_unit, axis=0)
                    f_FP_unit = f_FP_unit / np.linalg.norm(f_FP_unit)
                    params = self.build_params_inhomo(i, N, f_FP_unit)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        FDRs[i,j] = self.FDR_inhomo(params)
                        pbar.update(1)
                    
                else:
                    
                    if N*self.max_compare <= (n_neurons - 1):
                        comparisons = self.max_compare
                    else:
                        comparisons = np.floor((n_neurons - 1) / N)
                        
                        
                    FP_idx = sample(range(n_neurons - 1), N*comparisons)
                    FP_idx = [FP_idx[x:x + N] for x in range(0, len(FP_idx), N)]
                    
                    k_FDRs = []
                    for k in range(len(FP_idx)):
                        
                        f_FP_unit = np.sum(other_f_t_unit[FP_idx[k],:], axis=0)
                        f_FP_unit = f_FP_unit / np.linalg.norm(f_FP_unit)
                        params = self.build_params_inhomo(i, N, f_FP_unit)
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            k_FDRs.append(self.FDR_inhomo(params))
                        
                    FDRs[i,j] = np.mean(k_FDRs)
                    pbar.update(1)
        

        pbar.close()
                        
        return FDRs
    
    def pred_FDR_homo(self):
        
        n_neurons = self.f_t.shape[0]
        n_N = len(self.N)
        FDRs = np.zeros((n_neurons, n_N))
            
        # i = neuron index
        # j = N index 
        
        pbar = tqdm(total=n_neurons * n_N, 
                    desc='Calculating...')
        
        for j in range(n_N):
            
            N = self.N[j]
            
            for i in range(n_neurons):
                
                
                if N == float('inf'):
  
                    params = self.build_params_homo(i, N)
                    
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        FDRs[i,j] = self.FDR_homo(params)
                        pbar.update(1)
                    
                else:
                    
                    if N*self.max_compare <= (n_neurons - 1):
                        comparisons = self.max_compare
                    else:
                        comparisons = np.floor((n_neurons - 1) / N)
                        
                    FP_idx = sample(range(n_neurons - 1), N*comparisons)
                    FP_idx = [FP_idx[x:x + N] for x in range(0, len(FP_idx), N)]
                    
                    k_FDRs = []
                    for k in range(len(FP_idx)):
                        
                        params = self.build_params_homo(i, N)
                        
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            k_FDRs.append(self.FDR_homo(params))
                        
                    FDRs[i,j] = np.mean(k_FDRs)
                    pbar.update(1)
        

        pbar.close()
                        
        return FDRs
    
        

            
        