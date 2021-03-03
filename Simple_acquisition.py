# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:29:09 2021

@author: Maxime Farin

Example script to record data with Picoscope.
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.io import savemat
import scipy.signal as signal
mpl.rcParams['font.size'] = 18
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2

from Pypico import Pypico

# parameters
fs = 100e6 # Sampling frequency (Hz)
duration = 10e-6 # Duration of the recording (s)

# Data acquisition
p = Pypico() # Start Picoscope

try:
    p.set_channels([0], [0.01]) # channels, channel_range (V)
    p.set_trigger(trig_channel=0, threshold_mV=3, auto_trig_ms=1000) # Set trigger on channel A
    
    time_s, sig = p.simple_record([0], fs, duration) # Record time and signal on channel A

finally:
    p.stop_pico() # Stop Picoscope
    
fs = 1/(time_s[1] - time_s[0]) # actual sampling frequency

# Filtering
b, a = signal.butter(2, [2*1e6/fs], 'highpass')
sig_filt = signal.filtfilt(b, a, sig)  

# Save data
savemat('data.mat', {'time_s': time_s, 'sig' : sig_filt})

# Plot
fig = plt.figure()
plt.plot(time_s, sig_filt)
plt.xlabel("Time [s]")
plt.ylabel("Amplitude [mV]")
