# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 15:37:38 2021

@author: Maxime Farin

Use this script to measure the impulse response of a structure between frequencies f1 and f2.
Function chirp_and_rec: 
    - triggers Picoscope function generator to emit a chirp between frequencies f1 and f2.
      (The emitted chirp is recorded on channel B (connect external and channel B).)
    - correlates the measured signal on channel A with the emitted chirp (measured on channel B) 
    - returns the impulse response and associated time array.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 18
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2
from scipy.io import savemat
import scipy.signal as signal

from Pypico import Pypico

# Parameter definition
fs = 200e6 # Hz, sampling frequency
duration = 0.5e-3 # desired duration of the impulse response
f1 = 8e6 # Hz, start frequency of the chirp
f2 = 12e6 # Hz, stop frequency of the chirp


# Data acquisition
p = Pypico()
try:
    # set channels (A:0, B:1) and channel_range (V)
    p.set_channels([0, 1], [0.1, 1]) 
    # set trigger on channel A, with a 10 mV threshold
    p.set_trigger(trig_channel=0, threshold_mV=10) 
    
    # recording the impulse response on channel A
    time_s, sig = p.chirp_and_rec([0, 1], [1], fs, duration, f1, f2)
    
finally:
    # stop the Picoscope
    p.stop_pico()

# Recompute the sampling frequency (Picoscope may adjust it to a different but close value)
fs = 1/(time_s[1] - time_s[0])

# Filtering to remove noise
b, a = signal.butter(2, [2*f1/fs, 2*f2/fs], 'bandpass')
sig_filt = signal.filtfilt(b, a, sig)
     
# Save data
savemat('data.mat', {'time_s': time_s, 'sig': sig_filt})

# Plot
fig = plt.figure()
plt.plot(time_s, sig_filt, 'k')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")