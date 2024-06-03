Decisively quantify cross-contamination in spike-sorted electrophysiology data using DCISIv.

# Quick-start guide

Pip installable package coming soon, for now clone the repository and import the DCISIv class.  

```python
from DCISIv import DCISIv
```
This class requires two things to create a DCISIv object, (1) f_t: the firing frequencies of a group of clusters, and (2) ISI_v: their ISI violation rates. Firing frequencies can be a 1-D array for homogeneous firing calculations, or a 2-D n x m array for inhomogeneous firing calculations, where n is the number of neurons and m is the number of time points. We recommend using total average firing rates for homogeneous calculations when the data do not have a trial structure to average around. For trial-averaged data, sampling frequency is not particularly important. Use the largest PSTH bin size that still captures temporal variability in your data. 

```python
DCISIv_obj = DCISIv(f_t, ISI_v)
```


```python
f_t = np.load("Inagaki2019_PSTHs.npy")
t = np.arange(0, 6, 0.05)
```


![PSTHs](https://github.com/economolab/DCISIv/assets/60631663/2d1fbe5a-b462-4d52-bcb6-d687ce864201)

