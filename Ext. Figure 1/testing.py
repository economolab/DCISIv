# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:43:42 2024

@author: jpv88
"""

from scipy.stats import expon

rate = 10
refractory_period = 0.0025

effective_rate = rate / (1. - rate * refractory_period)
isi_generator = expon(scale=1. / effective_rate, loc=refractory_period)

# %%

import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)


r = isi_generator.rvs(size=1000)
ax.hist(r, density=True, bins='auto', histtype='stepfilled', alpha=0.2)
ax.legend(loc='best', frameon=False)
plt.show()