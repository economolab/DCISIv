# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:43:42 2024

@author: jpv88
"""
import numpy as np

from scipy.stats import expon

t_start = 0
t_stop = 10
rate = 10
refractory_period = 0.0025

effective_rate = rate / (1. - rate * refractory_period)
isi_generator = expon(scale=1. / effective_rate, loc=refractory_period)

n_expected_spikes = int(np.ceil(((t_stop - t_start) * rate)))
n_spikes_three_stds = int(np.ceil(n_expected_spikes + 3 * np.sqrt(n_expected_spikes)))

spikes = np.array([0])

# 3 STDs corresponds to 99.7%
n_spikes_three_stds = int(np.ceil(n_expected_spikes + 3 * np.sqrt(n_expected_spikes)))

# Continue until whole time range is covered
while spikes[-1] < t_stop:
    isi = isi_generator.rvs(size=n_spikes_three_stds)

    t_last_spike = spikes[-1]
    spikes = np.r_[spikes, t_last_spike + np.cumsum(isi)]

index_last_spike = spikes.searchsorted(t_stop)
spikes = spikes[:index_last_spike]

      

# %%

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)


r = isi_generator.rvs(size=1000)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()