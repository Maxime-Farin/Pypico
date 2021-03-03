# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:36:41 2021

@author: mfarin

Function library for python processing of signals

"""

import pickle
from math import sqrt, ceil, log
import numpy as np
from numpy.fft import rfft
import scipy.signal as signal


def read_byte_file(filename):
    '''
    Import data from a byte file
    '''
    data = []
    with open(filename, 'rb') as file:
        while 1:
            try:
                data.append(pickle.load(file))
            except EOFError:
                break
    return data


def decorrel_coef(sig1, sig2):
    '''
    Compute the decorrelation coefficient between two signals sig1 and sig2 (numpy arrays)
    '''
    return 1 - sum(sig1 * sig2) / sqrt(sum(sig1**2) * sum(sig2**2))


def fftrl(t_s, sig):
    '''
    Compute the real Fourier tranform of signal sig

    Parameters
    ----------
    t_s : Time in seconds
    sig : Signal (numpy array (time dimension * number of channels))

    Returns
    -------
    f : Frequency in Hz (numpy array)
    ft : Real Fourier transform of sig (numpy array (frequency dimension * number of channels))

    '''
    fs = 1/(t_s[2] - t_s[1]) # Sampling frequency (Hz)
    nfft = 2**(ceil(log(sig.shape[0]) / log(2))) # length of the fft
    window = signal.tukey(sig.shape[0], 0.1)# cosine taper window
    ft = rfft(sig * window, nfft) # rfft
    f = np.linspace(0, fs/2, len(ft)) # Frequency vector
    
    return (f, ft)


def sigfilt(sig, fs, freq, ftype='high', order=2):
    '''
    Simple filter of the signal 'sig' 

    Parameters
    ----------
    sig : Signal (numpy array (time dimension * channel number))
    fs : float, Sampling frequency (Hz)
    freq : float (for highpass and lowpass filter) or list of two float (for bandpass)
        Frequencies of the filter
    ftype : string, optional 
        Type of filter if only one frequency is given. Either 'high' or 'low'. The default is 'high'.
        If two frequencies are given, the filter is bandpass.
    order : int, optional
        Order of the filter. The default is 2.

    Returns
    -------
    sig_filt : Filtered signal (numpy array (time dimension * channel number))

    '''
    
    # Define butterworth filter parameters
    if isinstance(freq, int):
        if ftype == 'high':
            b, a = signal.butter(order, [2*freq/fs], 'highpass')
        elif ftype == 'low':
            b, a = signal.butter(order, [2*freq/fs], 'lowpass')
        else:
            raise Exception('Wrong filter type given !')
    elif isinstance(freq, list):
        b, a = signal.butter(order, [2*freq[0]/fs, 2*freq[1]/fs], 'bandpass')
    else:
        raise Exception('Wrong frequency type/number')
        
    # Filter signals
    sig_filt = np.zeros([sig.shape[0], sig.shape[1]], dtype=float)
    for n in range(sig.shape[1]):
        sig_filt[:, n] = signal.filtfilt(b, a, sig[:, n])
        
    return sig_filt