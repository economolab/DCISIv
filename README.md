Decisively assess cross-contamination in spike-sorted electrophysiology data using DCISIv.

Pre-print link (Peer-reviewed publication link coming soon)
https://www.biorxiv.org/content/10.1101/2023.12.21.572882v1

# Quick-start guide

To get up and running ASAP, follow these instructions. For more detailed demonstrations, see example.py. Pip installable package as well as implementations in phy and SpikeInterface coming soon, for now clone this repository, navigate to it in the terminal, and install the required packages using:
```python
pip install -r requirements.txt
```

Then, import the DCISIv class into your workspace:  

```python
from DCISIv import DCISIv
```
This class requires two things to create a DCISIv object, (1) f_t: the firing frequencies (aka firing rates) of a group of clusters, and (2) ISI_v: their ISI violation rates. Firing frequencies must be in units of Hz, and ISI violation rate must be raw, i.e. not already converted to a percentage. Firing frequencies can be a 1-D array for homogeneous firing calculations, or a 2-D n x m array for inhomogeneous firing calculations, where n is the number of neurons and m is the number of time points.
```python
DCISIv_obj = DCISIv(f_t, ISI_v)
```
This object can optionally be initialized with the following arguments: "N", "tau", "tau_c", with default values (1, float("inf")), 2.5, and 0. "N" is the number of assumed contaminant neurons. It can be a single value or sequence of values. "N" is not a particularly important parameter. "tau" is the neuronal absolute refractory period in ms and must be set correctly. Make sure "tau" here is the same value used to calculated ISI violations. We recommend the default value of 2.5 ms for mouse cortical neurons. "tau_c" is the spike sorter censor period in ms. The censor period is the window of time after a given spike that subsequent spikes are ignored, i.e. the shortest possible distance in time between two spikes in a cluster as enforced by the spike sorting algorithm. This value must also be set correctly to obtain the most accurate results.

```python
% Example object initializations 
DCISIv_obj = DCISIv(f_t, ISI_v, N=(2, 3, 4), tau=1, tau_c=0.25)
DCISIv_obj = DCISIv(f_t, ISI_v, N=2)
```

Results can be obtained from the object's results dictionary "res". 
```python
res = DCISIv_obj.res
```
This dictionary has 3 keys: "FDRs", "mean", and "median". The "mean" and "median" keys contain the population-level mean and median FDR. "FDRs" is a n x m dataframe where n is the number of clusters and m is the number of different assumed contaminant neuron counts (N) + 1. The column names correspond to the assumed N, except for the last column (mean) which is the average across all conditions. Values in the (mean) column of this dataframe are used to calculated the population level "mean" and "median" FDRs.
```python
print("Mean FDR = {:.3f}".format(res["mean"]))
print("Median FDR = {:.3f}".format(res["median"]))
```
```
Mean FDR = 0.139
Median FDR = 0.071
```

# Tips and best practices

- We DO NOT recommend using results obtained here as a per-cluster inclusion or exclusion criterion. As illustrated in Fig. 4 of the manuscript, FDR predicted using ISI violation rate is well-correlated with true FDR, however, the stochasticity of ISI violations means predictions for individual clusters can be highly inaccurate. This is especially true for lower-firing frequency clusters (<5 Hz) recorded for shorter time periods.
- We DO recommend using population-level mean and median predicted FDRs to assess cross-contamination across an entire recording session or dataset. Noise is averaged out across multiple clusters, producing highly accurate estimates of mean and median FDR.
- We DO recommend calculating results for each recording session and probe separately and then collating results afterward for a full dataset. This is because only clusters within the same recording session and probe should be used to obtain the most accurate estimates of f_FP over time.  
- We DO recommend using total average firing frequencies for homogeneous calculations when the data do not have a trial structure to average around. For trial-averaged data, sampling frequency is not particularly important. Use the largest PSTH bin size that still captures temporal variability in your data.
- To obtain tau_c, the spike sorting censor period, if you don't already know it: calculate ISIs for each cluster and then concatenate and plot as a histogram. The censor period should be obvious as a sharp and sudden drop off to 0 in the number of ISIs below a given value.

# TO-DO
- Add censor period to simulations in example.py
- Add restriction of f_FP to nearby electrode sites
- Make package pip installable
- phy and SpikeInterface implementations 
