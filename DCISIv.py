# -*- coding: utf-8 -*-
"""
Created on Tue May 21 11:31:41 2024

@author: jpv88
"""

import numpy as np


class calc_FDR():
    
    def __init__(self, f_t, ISI_v, N=(1, float('inf')), tau=2.5, tau_c=0):
        
        self.f_t = f_t
        self.ISI_v = ISI_v
        self.tau = tau 
        self.tau_c = tau_c
        self.N = N
        
        self.tau = self.tau*(10**-3)
        self.tau_c = self.tau_c*(10**-3)
        self.tau_e = self.tau - self.tau_c
        
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
                self.pred_FDR_homo()
            case 'inhomo':
                self.pred_FDR_inhomo()
        
        self.max_compare = 10

                
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
            
        def build_params(i, f_FP_unit):
            
            params = {}
            params['f_t'] = self.f_t[i,:]
            params['ISI_v'] = self.ISI_v[i]
            params['N'] = N
            params['tau_e'] = self.tau_e
            params['f_FP_unit'] = f_FP_unit
            
            return params
            
        for j in range(n_N):
            
            N = self.N[j]
            
            for i in range(n_neurons):
                
                other_idx = list(range(n_neurons))
                del other_idx[i]
                other_f_t_unit = f_t_unit[other_idx,:]
                
                if N == float('inf'):
                    f_FP_unit = np.sum(other_f_t_unit, axis=0)
                    f_FP_unit = f_FP_unit / np.linalg.norm(f_FP_unit)
                    params = build_params(i, f_FP_unit)
                    FDRs[i,j] = FDR_master(params)
                    
                else:
                    
                    if N*self.max_compare <= (n_neurons - 1):
                        
                        
                    
                
                
                
            
        #     other_idx = list(range(N))
        #     del other_idx[i]
        #     others_PSTH = PSTHs[other_idx,:]
        #     others_mean = np.mean(others_PSTH, axis=0)
        #     others_unit = others_mean/np.linalg.norm(others_mean)
            
        #     inf_FDR = FDR_master(ISI_viol[i], PSTHs[i,:], others_unit, N=float('inf'), tau=tau, tau_c=tau_c)
        #     if np.isnan(inf_FDR):
        #         inf_FDRs.append(1)
        #     else:
        #         inf_FDRs.append(inf_FDR)
            
        #     unit_FDRs = []
        #     for j in other_idx:
        #         temp = FDR_master(ISI_viol[i], PSTHs[i,:], PSTHs_unit[j,:], N=1, tau=tau, tau_c=tau_c)
        #         if np.isnan(temp):
        #             unit_FDRs.append(0.5)
        #         else:
        #             unit_FDRs.append(temp)
            
        #     one_FDRs.append(np.mean(unit_FDRs))
        
        # FDRs = [np.mean([el1, el2]) for el1, el2 in zip(one_FDRs, inf_FDRs)]
        
        return FDRs
    
        
    # the final equation
    def FDR_master(params):
        
        f_t = params['f_t']
        ISI_v = params['ISI_v']
        N = params['N'] 
        tau_e = params['tau_e']
        f_FP_unit = params['f_FP_unit']
    
        f_t_mag = np.linalg.norm(f_t)
        f_t_unit = f_t / f_t_mag
        f_t_avg = np.average(f_t)

        n = len(Rtot)
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
            
        