# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:04:50 2021

@author: Maxime Farin

Example script to record the impulse response of a structure excited by a chirp signal, every 30 seconds for 30 minutes

Setup: 
    - Connect external channel of the Picoscope with channel B and with the piezoelectric sensor connected to the investigated structure,
    - Connect the measuring sensor to channel A.

Every 30s, function chirp_and_rec will emit a chirp, correlate the emitted chirp (channel B) with measured signal (channel A) 
and return the impulse response of the structure in the frequency range of the chirp signal.

Time and impulse data are saved in 't_' + str(time of the experiment in seconds) + 's' files
Times of the experiment are saved in the file time_exp.txt

"""

from scipy.io import savemat
import scipy.signal as signal
import numpy as np
from time import time, sleep

from Pypico import Pypico

# Parameters
Total_duration = 30*60 # Total duration of the experiment (s)
Delta_t = 30 # time between two experiments (s)
fs = 125e6 # Hz, sampling frequency
duree = 1e-3 # duration of one impulse signal (s)
nb_acquisition = 10 # impulse is averaged over 10 acquisitions
f1 = 0.5e6 # Start frequency of the chirp (Hz)
f2 = 12e6 # Stop frequency of the chirp (Hz)

# File with the times of the measurements
foldername = 'Path/to/save/directory/'
with open(foldername + 'time_exp.txt', 'w') as file:
    file.write('time_s\n')

# Data acquisition
p = Pypico() # Starts Picoscope

try:
    p.set_channels([0, 1], [0.2, 1]) # channels, channel_range (V)
    p.set_trigger(trig_channel=0, threshold_mV=10) # set trigger on channel A
    
    t0 = time()
    elapsed_time = 0
    while elapsed_time < Total_duration:
        
        # recording nb_acquisition impulses
        temps, _ = p.chirp_and_rec([0,1], [1], fs, duree, f1, f2)
        sig = np.zeros([len(temps), nb_acquisition], dtype=float)
        for n in range(nb_acquisition):
            _, s = p.chirp_and_rec([0,1], [1], fs, duree, f1, f2) # record the impulse response on channel A
            sig[:, n] = s[:, 0]
    
        FS = 1/(temps[1] - temps[0]) # compute actual sampling frequency
        
        # Filtering
        b, a = signal.butter(2, [2*f1/FS, 2*f2/FS], 'bandpass')
        sig_filt = np.zeros([len(temps), nb_acquisition], dtype=float)
        for n in range(nb_acquisition):
            sig_filt[:, n] = signal.filtfilt(b, a, sig[:, n])
             
        # Average
        sig_mean = np.mean(sig_filt, axis=1)   
        
        # Save time and data
        elapsed_time = time() - t0
        dt = int(elapsed_time)
        filename = foldername + 't_' + str(dt) + 's'
        savemat(filename + '.mat', {'texp_s': dt, 'time_s': temps, 'data_mV': sig_mean})
        
        with open(foldername + 'time_exp.txt', 'a') as file:
            file.write(str(dt) + '\n')
        
        print("Elapsed time: " + str(round(dt/60, 1)) + ' min')
        
        sleep(Delta_t) # sleep until next experiment

finally:
    p.stop_pico() # Stops Picoscope
    