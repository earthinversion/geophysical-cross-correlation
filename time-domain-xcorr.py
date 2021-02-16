# ======================================================================================================================
# Title:  Compute time domain cross-correlation between two time series
# Author: Utpal Kumar
# Date:   16 Feb 2021
# ======================================================================================================================


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from synthetic_tests_lib import crosscorr, compute_shift
plt.style.use('seaborn')

cluster2 = ['BO.ABU', 'BO.NOK']

time_series = cluster2  # + cluster2 + cluster3
dirName = "data/"
fs = 748  # take 748 samples only
MR = len(time_series)
Y = np.zeros((MR, fs))
dictVals = {}
for ind, series in enumerate(time_series):
    filename = dirName + series + ".txt"
    df = pd.read_csv(filename, names=[
                     'time', 'U'], skiprows=1, delimiter='\s+')  # reading file as pandas dataframe to work easily

    # this code block is required as the different time series has not even sampling, so dealing with each data point separately comes handy
    # can be replaced by simply `yvalues = df['U]`
    yvalues = []
    for i in range(1, fs+1):
        val = df.loc[df['time'] == i]['U'].values[0]
        yvalues.append(val)

    dictVals[time_series[ind]] = yvalues


timeSeriesDf = pd.DataFrame(dictVals)
print(timeSeriesDf.head())


# plot time series
# simple `timeSeriesDf.plot()` is a quick way to plot
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(timeSeriesDf[time_series[0]], color='b', label=time_series[0])
ax[0].legend()
ax[1].plot(timeSeriesDf[time_series[1]], color='r', label=time_series[1])
ax[1].legend()
plt.savefig('data_viz.jpg', dpi=300, bbox_inches='tight')
plt.close('all')


calc_xcorr = 0
ind1 = 0
ind2 = 1
if calc_xcorr:

    d1, d2 = timeSeriesDf[time_series[ind1]], timeSeriesDf[time_series[ind2]]
    window = 10
    # lags = np.arange(-(fs), (fs), 1)  # uncontrained
    lags = np.arange(-(200), (200), 1)  # contrained
    rs = np.nan_to_num([crosscorr(d1, d2, lag) for lag in lags])

    print(
        "xcorr {}-{}".format(time_series[ind1], time_series[ind2]), lags[np.argmax(rs)], np.max(rs))

    fig, ax = plt.subplots(3, 1, figsize=(10, 6))
    ax[0].plot(timeSeriesDf[time_series[0]], color='b', label=time_series[0])
    ax[0].legend()
    ax[1].plot(timeSeriesDf[time_series[1]], color='r', label=time_series[1])
    ax[1].legend()

    ax[2].plot(lags, rs, color='k', label='time-domain xcorr')
    ax[2].legend()
    ax[2].axvline(x=lags[np.argmax(rs)])
    plt.savefig('data_xcorr.jpg', dpi=300, bbox_inches='tight')
    plt.close('all')

shift = compute_shift(
    timeSeriesDf[time_series[ind1]], timeSeriesDf[time_series[ind2]])

print(shift)
