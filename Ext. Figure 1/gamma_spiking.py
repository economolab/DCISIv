# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:43:42 2024

@author: jpv88
"""
import numpy as np

from scipy.optimize import root_scalar
from scipy.stats import expon, gamma, invgauss, lognorm
from scipy import integrate

from tqdm import tqdm

from scipy.special import gammainc, gammaincc


# %% Functions

def get_first_spike(isi_generator, t_start):
    
    return isi_generator.rvs() + t_start

def gen_spikes_gamma(CV, t_stop, rate, refractory_period):
    
    t_start = 0 
    shape_factor = 1/(CV**2)
    scale = (1/(shape_factor*rate)) - (refractory_period/shape_factor)
    isi_generator = gamma(a=shape_factor, scale=scale)

    spikes = np.array(get_first_spike(isi_generator, t_start))
    spikes = np.expand_dims(spikes, 0)
    
    # Continue until whole time range is covered
    while spikes[-1] < t_stop:
        isi = isi_generator.rvs()

        t_last_spike = spikes[-1]
        spikes = np.r_[spikes, t_last_spike + refractory_period + isi]

    index_last_spike = spikes.searchsorted(t_stop)
    spikes = spikes[:index_last_spike]
    
    return spikes

def sim_Fv_gamma(rate, FDR, CV, T=6, refractory_period=2.5, N=1000, 
                 out_refrac=2.5, neurons=1):
    
    F_v = np.zeros(N)
    Rtot = np.zeros(N)
    
    refractory_period = refractory_period * 0.001

    if neurons != float('inf'):
        neurons = int(neurons)
        
    out_refrac = refractory_period
    if neurons == float('inf'):
        out_refrac = 0
        neurons = 1

    Rout = rate*FDR
    Rin = rate - Rout
    
    for i in tqdm(range(N)):
        
        spks_in = gen_spikes_gamma(CV, T, Rin, refractory_period)
        spks_out = []
        for j in range(neurons):
            spks_out.append(gen_spikes_gamma(CV, T, Rout/neurons, out_refrac))
        
        spks_out = np.concatenate(spks_out)
        
        spks_tot = np.concatenate((spks_in, spks_out))
        spks_tot = np.sort(spks_tot)
        
        ISIs = np.diff(spks_tot)
        Nviols = sum(ISIs < refractory_period)
        F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
        Rtot[i] = len(spks_tot)/T
    
    return np.mean(F_v), np.mean(Rtot)

def gen_spikes_lognormal(CV, t_stop, rate, refractory_period):
    
    sigma = np.sqrt(np.log(CV**2 + 1))
    mu = np.log((1/rate) - refractory_period) - (sigma**2)/2
    t_start = 0 

    isi_generator = lognorm(s=sigma, scale=np.exp(mu))

    spikes = np.array(get_first_spike(isi_generator, t_start))
    spikes = np.expand_dims(spikes, 0)
    
    # Continue until whole time range is covered
    while spikes[-1] < t_stop:
        isi = isi_generator.rvs()

        t_last_spike = spikes[-1]
        spikes = np.r_[spikes, t_last_spike + refractory_period + isi]

    index_last_spike = spikes.searchsorted(t_stop)
    spikes = spikes[:index_last_spike]
    
    return spikes

def sim_Fv_lognormal(rate, FDR, CV, T=6, refractory_period=2.5, N=1000, 
                     out_refrac=2.5, neurons=1):
    
    F_v = np.zeros(N)
    Rtot = np.zeros(N)
    
    refractory_period = refractory_period * 0.001

    if neurons != float('inf'):
        neurons = int(neurons)
        
    out_refrac = refractory_period
    if neurons == float('inf'):
        out_refrac = 0
        neurons = 1

    Rout = rate*FDR
    Rin = rate - Rout
    
    for i in tqdm(range(N)):
        
        spks_in = gen_spikes_lognormal(CV, T, Rin, refractory_period)
        spks_out = []
        for j in range(neurons):
            spks_out.append(gen_spikes_lognormal(CV, T, Rout/neurons, out_refrac))
        
        spks_out = np.concatenate(spks_out)
        
        spks_tot = np.concatenate((spks_in, spks_out))
        spks_tot = np.sort(spks_tot)
        
        ISIs = np.diff(spks_tot)
        Nviols = sum(ISIs < refractory_period)
        F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
        Rtot[i] = len(spks_tot)/T
    
    return np.mean(F_v), np.mean(Rtot)

def gen_spikes_invgauss(CV, t_stop, rate, refractory_period):
    
    nu = (1-refractory_period*rate)/rate
    lam = nu/(CV**2)
    
    t_start = 0 

    isi_generator = invgauss(mu=nu/lam, loc=refractory_period, scale=lam)

    spikes = np.array(get_first_spike(isi_generator, t_start))
    spikes = np.expand_dims(spikes, 0)
    
    # Continue until whole time range is covered
    while spikes[-1] < t_stop:
        isi = isi_generator.rvs()

        t_last_spike = spikes[-1]
        spikes = np.r_[spikes, t_last_spike + refractory_period + isi]

    index_last_spike = spikes.searchsorted(t_stop)
    spikes = spikes[:index_last_spike]
    
    return spikes

def sim_Fv_invgauss(rate, FDR, CV, T=6, refractory_period=2.5, N=1000, 
                     out_refrac=2.5, neurons=1):
    
    F_v = np.zeros(N)
    Rtot = np.zeros(N)
    
    refractory_period = refractory_period * 0.001

    if neurons != float('inf'):
        neurons = int(neurons)
        
    out_refrac = refractory_period
    if neurons == float('inf'):
        out_refrac = 0
        neurons = 1

    Rout = rate*FDR
    Rin = rate - Rout
    
    for i in tqdm(range(N)):
        
        spks_in = gen_spikes_invgauss(CV, T, Rin, refractory_period)
        spks_out = []
        for j in range(neurons):
            spks_out.append(gen_spikes_invgauss(CV, T, Rout/neurons, out_refrac))
        
        spks_out = np.concatenate(spks_out)
        
        spks_tot = np.concatenate((spks_in, spks_out))
        spks_tot = np.sort(spks_tot)
        
        ISIs = np.diff(spks_tot)
        Nviols = sum(ISIs < refractory_period)
        F_v[i] = Nviols/len(spks_tot) if len(spks_tot) != 0 else 0
        Rtot[i] = len(spks_tot)/T
    
    return np.mean(F_v), np.mean(Rtot)


# def get_first_spike_equilibrium(rate, isi_generator, t_start, t_stop, shape_factor=None):
#         """
#         Return a numerically drawn sample of the p.d.f of the first spike.

#         By solving:
#         x = integral(c.d.f(t) from 0 to t),
#         where x is drawn from a uniform distribution.
#         """
        
#         # random float in the half-open interval [0.0, 1.0)
#         random_uniform = np.random.random()
        
#         # scipy routine to find the root of a scalar function
#         equation_solver = root_scalar
        
#         if shape_factor == None:
            
#             def cdf_first_spike_equilibrium(time):
#                 """
#                 Integral over the p.d.f. of the first spike which is:
#                 p(t) = rate * survival-function(t) * Heaviside(t).
#                 See Bouss (2020).
        
#                 The parameter time is a magnitude of a time value given in seconds.
#                 """
#                 return rate * integrate.quad(isi_generator.sf, 0., time)[0]
            
#         else: 
        
#             def cdf_first_spike_equilibrium(time):
#                 """
#                 The parameter time is a magnitude of a time value given in seconds.
#                 """
#                 if time < 0.:
#                     return 0.
#                 return rate * time * \
#                     gammaincc(shape_factor,
#                               shape_factor*rate*time)\
#                     + gammainc(shape_factor+1.,
#                                shape_factor*rate*time)
        
#         def function_to_solve(time):
#             """
#             # integral(c.d.f(t) from 0 to t) - random-number-x)
#             """
#             return cdf_first_spike_equilibrium(time) - random_uniform

#         def derivative_of_function_to_solve(time):
#             """
#             derivative of the c.d.f, which is rate times
#             the survival function
#             """
#             return rate * isi_generator.sf(time)

#         # Initial guess is solution for Poisson process
#         initial_guess = -np.log(1.-random_uniform)/rate
#         duration = t_stop-t_start
#         # limits_for_first_spike = (0, duration)
#         limits_for_first_spike = (0, 10/rate)


#         # test if solution for first spike is inside the boundaries. If not
#         # return t_stop of the spike train.
#         # if cdf_first_spike_equilibrium(duration) <= random_uniform:
#         #     return t_stop
        
#         method = 'newton'
#         if shape_factor != None:
#             method = 'brentq'

#         non_shifted_position_of_first_spike = equation_solver(
#                 function_to_solve,
#                 x0=initial_guess,
#                 bracket=limits_for_first_spike,
#                 fprime=derivative_of_function_to_solve,
#                 method=method
#             ).root

#         return non_shifted_position_of_first_spike + t_start

# def get_first_spike(rate, refractory_period, t_start):

#         effective_rate = rate / (1. - rate * refractory_period)
        
#         # the case with dead time
#         random_uniform = np.random.random()
#         if random_uniform <= rate * refractory_period:
#             return random_uniform / rate + t_start

#         return (np.log(1. - rate * refractory_period)
#                 - np.log(1. - random_uniform)
#                 ) / effective_rate + refractory_period

# def gen_spikes(isi_generator, t_start, t_stop, rate, refractory_period):
    
#     n_expected_spikes = int(np.ceil(((t_stop - t_start) * rate)))
#     n_spikes_three_stds = int(np.ceil(n_expected_spikes + 3 * np.sqrt(n_expected_spikes)))

#     spikes = np.array(get_first_spike(rate, refractory_period, t_start))
#     spikes = np.expand_dims(spikes, 0)

#     # 3 STDs corresponds to 99.7%, aka 99.7% of the time, you should only have to 
#     # generate ISIs once to fill the requested interval of time
#     n_spikes_three_stds = int(np.ceil(n_expected_spikes + 3 * np.sqrt(n_expected_spikes)))

#     # Continue until whole time range is covered
#     while spikes[-1] < t_stop:
#         isi = isi_generator.rvs(size=n_spikes_three_stds)

#         t_last_spike = spikes[-1]
#         spikes = np.r_[spikes, t_last_spike + np.cumsum(isi)]

#     index_last_spike = spikes.searchsorted(t_stop)
#     spikes = spikes[:index_last_spike]
    
#     return spikes

    

# # %% Poisson spike generation

# # some spike generation time is lost to refractoriness, this effective_rate is 
# # a bit higher than the input rate to account for that and get the actual 
# # number of requested spikes
# effective_rate = rate / (1. - rate * refractory_period)
# isi_generator = expon(scale=1. / effective_rate, loc=refractory_period)

# num_spikes = []
# for _ in range(1000):
#     spikes = gen_spikes(isi_generator, t_start, t_stop, rate, refractory_period)
#     num_spikes.append(len(spikes))
    
# print(np.mean(num_spikes))


# %%


CV = 1
spikes = gen_spikes_invgauss(CV, 1000, 10, 0.0025)




# %%

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)

t_start = 0 
rate = 10
refractory_period = 0.0025
CV = 4

sigma = np.sqrt(np.log(CV**2 + 1))
mu = np.log((1/rate) - refractory_period) - (sigma**2)/2
isi_generator = lognorm(s=sigma, scale=np.exp(mu))

x = np.linspace(isi_generator.ppf(0.01),
                isi_generator.ppf(0.99), 100)

ax.plot(x, isi_generator.pdf(x),
       'r-', lw=2, alpha=0.6)


r = isi_generator.rvs(size=1000)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
plt.show()