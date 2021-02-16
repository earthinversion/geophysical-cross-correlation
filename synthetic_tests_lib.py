# ======================================================================================================================
# Title:  Compute time domain cross-correlation between two time series
# Author: Utpal Kumar
# Date:   16 Feb 2021
# ======================================================================================================================

import numpy as np
from numpy.fft import fft, ifft, fft2, ifft2, fftshift

# Time lagged cross correlation


def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Shifted data filled with NaNs 

    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length
    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))


def cross_correlation_using_fft(x, y):
    f1 = fft(x)
    f2 = fft(np.flipud(y))
    cc = np.real(ifft(f1 * f2))
    return fftshift(cc)

# shift 0 means that y starts 'shift' time steps before x # shift &gt; 0 means that y starts 'shift' time steps after x


def compute_shift(x, y):
    assert len(x) == len(y)
    c = cross_correlation_using_fft(x, y)
    assert len(c) == len(x)
    zero_index = int(len(x) / 2) - 1
    shift = zero_index - np.argmax(c)
    return shift


def noisecorr(tr1, tr2,dt=1, window_length=3600., overlap=0.5,\
              onebit=False,whiten=True, waterlevel=1e-10,cos_taper=True,\
              taper_width=0.05):
    
   
   
    win_samples=int(window_length/dt)
    data1 = tr1.data
    data2 = tr2.data
    no_windows=int((len(data1)-win_samples)/((1-overlap)*win_samples))+1
    freq=np.fft.rfftfreq(win_samples,dt)
    
    if cos_taper:
        taper = cosine_taper(win_samples,p=taper_width)     

    # loop for all time windows
    for i in range(no_windows):
        window0=int(i*(1-overlap)*win_samples)
        window1=window0+win_samples
        d1=data1[window0:window1]
        d2=data2[window0:window1]
        d1 = detrend(d1,type='constant')
        d2 = detrend(d2,type='constant')
        
        if onebit:
            # 1 bit normalization - doesn't work very convincingly like that(?)
            d1=np.sign(d1)
            d2=np.sign(d2)
        
        if cos_taper: # tapering in the time domain
            d1*=taper
            d2*=taper
        
        # time -> freqency domain
        D1=np.fft.rfft(d1)
        D2=np.fft.rfft(d2)
        
        if whiten:
            #D1=np.exp(1j * np.angle(D1)) # too slow
            #D2=np.exp(1j * np.angle(D2))
            D1/=np.abs(D1)+waterlevel
            D2/=np.abs(D2)+waterlevel
            
        # actual correlation in the frequency domain
        CORR=np.conj(D1)*D2

        # summing all time windows
        if i==0:
            SUMCORR=CORR
        else:
            SUMCORR+=CORR
    
    #freq=freq[(freq>freqmin) & (freq<=freqmax)]
    SUMCORR /= float(no_windows)
    return freq, SUMCORR