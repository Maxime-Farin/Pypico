# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 09:38:56 2021

@author: Maxime Farin

Example script to plot the decorrelation over time of the impulse data acquired with Long_acquisition_pico.py
Decorrelation is computed over successive windows
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.size'] = 18
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Helvetica'
mpl.rcParams['lines.linewidth'] = 2
from scipy.io import loadmat
import numpy as np

from Func_lib_corrosion import decorrel_coef

import os
os.chdir('Path/to/save/directory')


# Import times of the experiments
with open('time_exp.txt', 'r') as file:
    t_s = file.readlines()
t_s = np.array([int(n) for n in t_s[1:]]) # time in s of the experiments

n_manipes = len(t_s) # number of experiments

result_files = {n:'t_' + str(t_s[n]) + 's' for n in range(n_manipes)} # filename of the experiments

# import data
temp = loadmat(result_files[1])
temps = temp['time_s'][0]
A = np.zeros([len(temps), n_manipes], dtype=float)
for k, v in result_files.items():
    temp = loadmat(v)
    A[:, k] = temp['data_mV'][0]
    
fs = 1/(temps[1]-temps[0]) # Sampling frequency

start_time = 2.125e-5 # Start time (s) in the coda signal
start_index = int(fs*start_time) # start index in the coda signal
dt_window = 10e-6 # duration of the correlated window (s)
nb_window = 10 # number of correlated windows
t0_window = np.linspace(start_time, start_time + (nb_window - 1)*dt_window, nb_window) # start times of the correlated window

data_A_ref = A[:, 0] # Reference impulse data

# Compute decorrelation coefficient
decorrel_coef_A = np.zeros([n_manipes, nb_window], dtype = float)
for ii in range(n_manipes):
    for jj in range(nb_window):
        ind1, ind2 = int(t0_window[jj] * fs), int((t0_window[jj] + dt_window)*fs)
        # Compute decorrellation coef on each channel, for each time and window
        decorrel_coef_A[ii, jj] = decorrel_coef(data_A_ref[ind1 : ind2], A[ind1 : ind2, ii]) 
        
# Plot data
fig = plt.figure(figsize = (15, 8))
ax1 = plt.subplot(111)
p1 = plt.plot(t_s/60, decorrel_coef_A);
plt.ylim(0, 2)
plt.legend(p1, map(lambda x: str(round(x*1e6, 2)) + ' µs', t0_window - start_time), loc='best', ncol = 4)
plt.title("Decorrelation coef.", fontweight='bold')
ax1.tick_params(direction='in')
plt.xlabel("time [min]")
fig.tight_layout()