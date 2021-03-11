# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 18:04:06 2021

@author: maxime farin

class pypico
"""

import ctypes
import numpy as np
import scipy.signal as signal
from picosdk.ps5000a import ps5000a as ps
from picosdk.functions import adc2mV, assert_pico_ok, mV2adc
from numpy.fft import rfft, fft, fft2, irfft
from time import sleep
from math import ceil, log

class Pypico():
    
    '''
    ############
    Class Pypico
    ############
    
    Define a Pypico object to communicate with Picoscope 5000 series, record signals and impulse response
    
    
    ############
    Example script to record the impulse response of the media between frequencies f1 and f2
    ############
    from Pypico_class import Pypico

    fs = 200e6 # Hz, sampling frequency
    duration = 0.5e-3 # Recording duration (s)
    f1 = 8e6
    f2 = 12e6
    nb_acquisition = 10

    p = Pypico() # Starts the Picoscope
    try:
        p.set_channels([0,1], [0.1, 1]) # channels, channel_range (V)
        p.set_trigger(trig_channel=0, threshold_mV=10, auto_trig_ms=1000) # Set trigger on channel 0, with a threshold amplitude at 10 mV, and an auto trig after 1 s (if nothing happens)
        
        # recording nb_acquisition impulses
        t_s, _ = p.chirp_and_rec([0,1], [1], fs, duration, f1, f2) # time in seconds 
        sig = np.zeros([len(t_s), nb_acquisition], dtype=float)
        for n in range(nb_acquisition):
            _, s = p.chirp_and_rec([0,1], [1], fs, duration, f1, f2) # record impulse 
            sig[:, n] = s[:, 0]
        
    finally:
        p.stop_pico() # Stops the Picoscope
    ###########   
        
    '''
    
    
    def __init__(self, res_bits=14):
        """
        Starts Picoscope with indicated bit rate
        at object Pypico definition

        Parameters
        ----------
        res_bits : int, optional
            Bit rate of the picoscope. The default is 14.

        Returns
        -------
        None.

        """
        # number/letters of the channels
        self.channels_numbers = {0:'A', 1:'B', 2:'C', 3:'D'}
        
        # Resolution set to 14 Bit by default
        self.res_bits = res_bits
        if self.res_bits not in [8, 12, 14, 15, 16]:
            raise Exception("Possible bit resolutions are [8, 12, 14, 15, 16]")
            
        # Create chandle and status ready for use
        self.chandle = ctypes.c_int16()
        self.status = {}
        
        # Open 5000 series PicoScope
        resolution = ps.PS5000A_DEVICE_RESOLUTION["PS5000A_DR_" + str(self.res_bits) + "BIT"]   
        # Returns handle to chandle for use in future API functions
        self.status["openunit"] = ps.ps5000aOpenUnit(ctypes.byref(self.chandle), None, resolution)

        # Open Picoscope
        print("Starting Picoscope...")
        try:
            assert_pico_ok(self.status["openunit"])
        except: # PicoNotOkError:
            powerStatus = self.status["openunit"]
            if powerStatus == 286:
                self.status["changePowerSource"] = ps.ps5000aChangePowerSource(self.chandle, powerStatus)
            elif powerStatus == 282:
                self.status["changePowerSource"] = ps.ps5000aChangePowerSource(self.chandle, powerStatus)
            else:
                raise
            assert_pico_ok(self.status["changePowerSource"])
            
        sleep(0.1)
        
        
        
    def set_channels(self, channels=[0,], channel_range=[10,]):
        '''
        Function set_channels:
            Sets up channels and volt ranges for these channels.

        Parameters
        ----------
        channels : list of int, optional
            List of channel numbers one intend to use for the measurement (0:A, 1:B, 2:C, 3:D)
            The default is 0 (Channel A).
        
        channel_range : list of float, optional
            Volt range of the channels (V) (list with same size as channels)
            Recorded values are varying between +/- channel_range
            Possible values are in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20]. 
            The default is 10V.

        Returns
        -------
        None.

        '''
        if isinstance(channels, int):
            channels = [channels]
        if isinstance(channel_range, int):
            channel_range = [channel_range]
        
        self.channels = channels
        for c in self.channels:
            if c not in [0, 1, 2, 3]:
                raise Exception('Possible channels are [0, 1, 2, 3] - (0:A, 1:B, 2:C, 3:D)')
        self.channel_range = channel_range  
        for c in self.channel_range:
            if c not in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20.]:
                raise Exception('Possible ranges are [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1., 2., 5., 10., 20.] V')
        if len(self.channels) != len(self.channel_range):
            raise Exception('Please specify range for each channel')
            
        for i, cr in enumerate(self.channel_range):
            if cr < 1.0:
                self.channel_range[i] = ps.PS5000A_RANGE["PS5000A_" + str(int(cr*1000)) + "MV"]
            else:
                self.channel_range[i] = ps.PS5000A_RANGE["PS5000A_" + str(int(cr)) + "V"]
            
        coupling_type = ps.PS5000A_COUPLING["PS5000A_DC"]
        for ch, cr in zip(self.channels, self.channel_range):
            chan = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_" + self.channels_numbers[ch]]
            # enabled = 1
            # analogue offset = 0 V
            self.status["setCh" + self.channels_numbers[ch]] = ps.ps5000aSetChannel(self.chandle, chan, 1, coupling_type, cr, 0)
            assert_pico_ok(self.status["setCh" + self.channels_numbers[ch]]) 
            
            
        # Find maximum amplitude count value
        # handle = chandle
        # pointer to value = ctypes.byref(maxADC)
        self.maxADC = ctypes.c_int16()
        self.status["maximumValue"] = ps.ps5000aMaximumValue(self.chandle, ctypes.byref(self.maxADC))
        assert_pico_ok(self.status["maximumValue"])
            
        print('Channels set.')
        
        
    def set_trigger(self, trig_channel=0, threshold_mV=100, auto_trig_ms=1000):
        '''
        Function set_trigger:
            Sets the trigger on one channel to start recording signal.
        
        Parameters
        ----------
        trig_channel : int = 0, 1, 2 or 3 optional
            The channel to trig on. The default is '0:A'.
        threshold_mV : int, optional
            Trigger threshold in mV. The default is 100.
        auto_trig_ms : int, optional
            Time after which the channel triggers automatically (ms). The default is 1000.

        Returns
        -------
        None.

        '''
        # Set up trigger
        # handle = chandle
        # enabled = 1
        source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_" + self.channels_numbers[trig_channel]]
        threshold = int(mV2adc(threshold_mV, self.channel_range[trig_channel], self.maxADC)) # trigger at 100 mV (mV2adc converts mV to counts)
        # direction = PS5000A_RISING = 2
        # delay = 0 s
        # auto Trigger = 0 ms
        self.status["trigger"] = ps.ps5000aSetSimpleTrigger(self.chandle, 1, source, threshold, 2, 0, auto_trig_ms)
        assert_pico_ok(self.status["trigger"])
        
        print('Trigger set.')
        
        
        
    def simple_record(self, channels, fs, duration, pretrig=0.1):
        '''
        Function simple_record:
            Records the signal on channels "channels", with sampling frequency "fs", for duration "duration", with a pretrig percentage of duration

        Parameters
        ----------
        channels : list of int
            Channels on which to measure signal.
        fs : int
            Sampling frequency (Hz)
        duration : float
            Duration (s) of the recorded signal, after pretrig.
        pretrig : float, optional
            Percentage of duration to record before trigger (%). The default is 0.1%.

        Returns
        -------
        t_s : numpy array of float
            time in seconds (max time: (1 + pretrig) * duration), length: (1 + pretrig) * duration) * fs 
        sig : numpy array of float
            signal in mV on each channels (len), same length as time.

        '''
        # Start acquisition
        self._runblock(fs, duration, pretrig)
        
        # Check for data collection to finish using ps5000aIsReady
        ready = ctypes.c_int16(0)
        check = ctypes.c_int16(0)
        while ready.value == check.value:
            self.status["isReady"] = ps.ps5000aIsReady(self.chandle, ctypes.byref(ready))
    
        # Collect data from buffers
        t_s, sig = self._collect_data(channels)
        
        return (t_s, sig)
    
    
    
    def chirp_and_rec(self, channels, chirp_channel, fs, duration, f1, f2, nb_steps=100):
        '''
        Function chirp_and_rec:
            Triggers Picoscope emission of a chirp on the external channel.
            The chirp generated by the Picoscope is not a standard chirp function as could be defined with python.
            The phase is not swept linearly, instead, the frequency changes by steps (the number of steps is defined by nb_steps).
            Therefore, in order to decorrelate the recorded signals from chirp emission, one have to record the emitted chirp on the channel "chirp_channel".
            The signal data of the structure excited with the chirp is collected on channels "channels"
            The function will correlate the chirp recorded on channel "chirp_channel" with the signals recorded on channels "channels"
            and return the impulse response of the structure for each "channels"

        Parameters
        ----------
        channels : list of ints
            Channels on which one want to measure the impulse response of the structure excited with the chirp
        chirp_channel : int or [int]
            Channel on which the chirp is recorded.
        fs : float
            Desired sampling frequency (Hz) (ATTENTION: the Picoscope can adjust the frequency to a given different (but close) value)
            One can retrieve the real sampling frequency using fs = 1/(t_s[1] - t_s[0])
        duration : float
            Duration of the impulse response (and duration of the emitted chirp signal) (s)
        f1 : float
            Start frequency of the chirp (Hz)
        f2 : float
            Stop frequency of the chirp (Hz)
        nb_steps : int, optional
            Number of frequency steps between f1 and f2 in the chirp. The default is 100.

        Returns
        -------
        t_s : numpy array (dimension [fs*duration])
            Time in seconds for the impulse responses.
        impulse : numpy array (dimensions [fs*duration, number of channels])
            Normalized impulse responses on each channels.

        '''
        mes_ok = 1
        while mes_ok:
            
            try:
                # Define output chirp function
                self._define_chirp(f1, f2, duration, nb_steps)
                
                # Start acquisition
                self._runblock(fs, duration*3, 0.1) # record a longer signal to be able to catch a full chirp in the recorded data (chirps are continuously swept)
                
                # Trigger emission
                self.status["triggerEmission"] = ps.ps5000aSigGenSoftwareControl(self.chandle, 1)
                assert_pico_ok(self.status["triggerEmission"])
                
                # Check for data collection to finish using ps5000aIsReady
                ready = ctypes.c_int16(0)
                check = ctypes.c_int16(0)
                while ready.value == check.value:
                    self.status["isReady"] = ps.ps5000aIsReady(self.chandle, ctypes.byref(ready))
    
                # Collect data from buffers
                temps, sig = self._collect_data(channels)
                #return (temps, sig)
                
                # Find chirp in data and compute the impulse response (deconvolved from chirp) for each channels
                t_s, impulse = self._compute_impulse(temps, sig, channels, chirp_channel, duration, f1)
                
                mes_ok = 0
                
            except:
                print("Chirp not found, retrying...")
                
        return (t_s, impulse)
        
    
    
    def stop_pico(self):
        '''
        Function stop_pico:
            Stops the picoscope
        
        !!! ATTENTION !!!
        It is advised to run your code after a "try:" and to call function stop_pico() in a "finally:".
        E.g.:
            
            p = Pypico() # starts Picoscope
            try:
                p.set_channels([0,1], [0.2, 1]) # channels, channel_range (V) # set channels
                p.set_trigger(trig_channel=0, threshold_mV=10, auto_trig_ms=1000) # set trigger
                
                t, s = p.chirp_and_rec([0,1], [1], fs, duree, f1, f2, n_steps) # send chirp and record impulses
                
            finally:
                p.stop_pico() # Stops Picoscope even if something goes wrong
                

        Doing so, the Picoscope will always be closed, even if an exception raises (even after keyboard interrupt!).
        (otherwise the Picoscope may crash and you will have to restart the kernel)

        Returns
        -------
        None.

        '''
        # Stop the scope
        print("Stop Picoscope...")
        # handle = chandle
        self.status["stop"] = ps.ps5000aStop(self.chandle)
        assert_pico_ok(self.status["stop"])
        
        # Close unit Disconnect the scope
        # handle = chandle
        self.status["close"]=ps.ps5000aCloseUnit(self.chandle)
        assert_pico_ok(self.status["close"])
        
        # display status returns
        
        
        print(" ")
        print('Picoscope closed.')
        print(" ")
        #print('Status: ')
        #print(self.status)
        
        
        
    
    # Utility functions
    
    def _runblock(self, fs, duration, pretrig):
        '''
        Utility function _runblock:
            Set timebase (sampling frequency) and starts data recording

        Parameters
        ----------
        fs : float
            Desired sampling frequency (Hz).
        duration : float
            Record duration (s)
        pretrig : float
            sets the duration of recording before trigger (pretrig duration = pretrig*duration)

        Returns
        -------
        None.

        '''
        #print("Start recording...")
        
        # Set number of pre and post trigger samples to be collected
        preTriggerSamples = int(duration * pretrig * fs)
        postTriggerSamples = int(duration * fs)
        self.maxSamples = preTriggerSamples + postTriggerSamples
        
        # Get timebase information
        # handle = chandle
        timebase = round(125e6/fs + 2) # compute the timebase (at 14 bits : timebase = 125e6*Dt+2) # to do
        # noSamples = maxSamples
        # pointer to timeIntervalNanoseconds = ctypes.byref(timeIntervalns)
        # pointer to maxSamples = ctypes.byref(returnedMaxSamples)
        # segment index = 0
        self.timeIntervalns = ctypes.c_float()
        returnedMaxSamples = ctypes.c_int32()
        self.status["getTimebase2"] = ps.ps5000aGetTimebase2(self.chandle, timebase, self.maxSamples, ctypes.byref(self.timeIntervalns), ctypes.byref(returnedMaxSamples), 0)
        assert_pico_ok(self.status["getTimebase2"])
        
        #FS = 1/(self.timeIntervalns.value*1e-9) # real sampling frequency
        
        # Run block capture
        # handle = chandle
        # number of pre-trigger samples = preTriggerSamples
        # number of post-trigger samples = PostTriggerSamples
        # timebase = (see Programmer's guide for mre information on timebases)
        # time indisposed ms = None
        # segment index = 0
        # lpReady = None (using ps5000aIsReady rather than ps5000aBlockReady)
        # pParameter = None
        self.status["runBlock"] = ps.ps5000aRunBlock(self.chandle, preTriggerSamples, postTriggerSamples, timebase, None, 0, None, None)
        assert_pico_ok(self.status["runBlock"]) 

    
    
    def _collect_data(self, channels):
        '''
        Utility function _collect_data:
            Define buffers and collect data 

        Parameters
        ----------
        channels : list of ints
            Channels numbers.

        Returns
        -------
        time_s : numpy array
            Time in seconds.
        signal_mV : numpy array
            signal in mV on each channel.

        '''
        #print("Collecting data...")
        
        # Create buffers ready for assigning pointers for data collection
        bufferMax = {}
        bufferMin = {}
        for ch in channels:
            bufferMax[ch] = (ctypes.c_int16 * self.maxSamples)()
            bufferMin[ch] = (ctypes.c_int16 * self.maxSamples)() # used for downsampling which isn't in the scope of this example
        
            # Set data buffer location for data collection from channel A
            # handle = chandle
            source = ps.PS5000A_CHANNEL["PS5000A_CHANNEL_" + self.channels_numbers[ch]]
            # pointer to buffer max = ctypes.byref(bufferAMax)
            # pointer to buffer min = ctypes.byref(bufferAMin)
            # buffer length = maxSamples
            # segment index = 0
            # ratio mode = PS5000A_RATIO_MODE_NONE = 0
            self.status["setDataBuffers" + self.channels_numbers[ch]] = ps.ps5000aSetDataBuffers(self.chandle, source, ctypes.byref(bufferMax[ch]), ctypes.byref(bufferMin[ch]), self.maxSamples, 0, 0)
            assert_pico_ok(self.status["setDataBuffers" + self.channels_numbers[ch]])
            
        # create overflow location
        overflow = ctypes.c_int16()
        # create converted type maxSamples
        cmaxSamples = ctypes.c_int32(self.maxSamples)
        
        # Retried data from scope to buffers assigned above
        # handle = chandle
        # start index = 0
        # pointer to number of samples = ctypes.byref(cmaxSamples)
        # downsample ratio = 0
        # downsample ratio mode = PS5000A_RATIO_MODE_NONE
        # pointer to overflow = ctypes.byref(overflow))
        self.status["getValues"] = ps.ps5000aGetValues(self.chandle, 0, ctypes.byref(cmaxSamples), 0, 0, 0, ctypes.byref(overflow))
        assert_pico_ok(self.status["getValues"])
        
        # create array to collect signal data (mV)
        signal_mV = np.zeros([cmaxSamples.value, len(channels)], dtype=float) 
        for ch in channels:
            # convert ADC counts data to mV
            signal_mV[:, ch] = adc2mV(bufferMax[ch], self.channel_range[ch], self.maxADC)
        
        # Create time data (s)
        time_s = np.linspace(0, cmaxSamples.value * self.timeIntervalns.value/1e9, cmaxSamples.value)
    
        #print("Data collected!")
        return (time_s, signal_mV)
    
    

        
    def _define_chirp(self, f1, f2, duration, nb_steps):
        '''
        Utility function _define_chirp:
            Set the function generator of the picoscope to generate chirps between required frequencies and of required duration.
            This function is used in the chirp_and_rec() function.

        Parameters
        ----------
        f1 : float
            Start frequency (Hz)
        f2 : float
            Stop frequency (Hz).
        duration : float
            Duration of the chirp (s).
        nb_steps : int
            Number of frequency steps between f1 and f2.

        Returns
        -------
        None.

        '''
        # handle = chandle
        # offsetVoltage = 0
        # pkToPk = 2000000
        # waveType = ctypes.c_int16(0) = PS5000A_SINE
        # startFrequency = 10 kHz
        # stopFrequency = 10 kHz
        # increment = 0
        # dwellTime = 1
        # sweepType = ctypes.c_int16(1) = PS5000A_UP
        # operation = 0
        shots = ctypes.c_uint32(0)
        sweeps = ctypes.c_uint32(50) # 50 sweeps of the chirp
        # triggerType = ctypes.c_int16(0) = PS5000a_SIGGEN_RISING
        # triggerSource = ctypes.c_int16(0) = P5000a_SIGGEN_NONE
        # extInThreshold = 1
        wavetype = ctypes.c_int32(0)
        sweepType = ctypes.c_int32(0)
        triggerType = ctypes.c_int32(0)
        triggerSource = ctypes.c_int32(4) # software trigger
        
        Delta_F = (f2 - f1) / nb_steps # frequency step between two frequency of the chirp (Hz)
        Delta_T = duration / nb_steps # time step (s)
        
        # Maximum output voltage is +/-2V for the picoscope, 1.9V is used to set the channel range to 2V and have a good dynamics on the channel that record the chirp
        self.status["setSigGenBuiltInV2"] = ps.ps5000aSetSigGenBuiltInV2(self.chandle, 0, 1900000, wavetype, f1, f2, Delta_F, Delta_T, sweepType, 0, shots, sweeps, triggerType, triggerSource, 0)    
        assert_pico_ok(self.status["setSigGenBuiltInV2"])
        
        
        
    
    def _compute_impulse(self, temps, sig, channels, chirp_channel, duration, f1):
        '''
        Utility function _compute_impulse:
            The function generator of the picoscope sweeps chirps continuously once triggered.
            This function finds the beginning and end indices of one chirp in the recorded signal (by tracking phase changes)
            and convolve the signals measured on channels "channels" with the chirp measured on channel "chirp_channel"
            in order to compute the impulse responses.
            This function is used in the chirp_and_rec() function.

        Parameters
        ----------
        temps : numpy array
            Time in seconds for the recorded signals.
        sig : numpy array
            Recorded signals in mV measured on every set channels.
        channels : list of int
            Channels on which we would like to measure the impulse response.
        chirp_channel : int or [int]
            Channel on which the emitted chirp is measured.
        duration : float
            Duration of the impulse response.
        f1 : float
            Start frequency of the emitted chirp.

        Returns
        -------
        t_s : numpy array (dimension [fs*duration])
            Time in seconds for the impulse responses.
        impulse : numpy array (dimensions [fs*duration, number of channels])
            Normalized impulse responses on each channels.

        '''
        fs = 1/(temps[2] - temps[1]) # actual sampling frequency (Hz)

        ## define the chirp signal and the signals measured on other channels
        if isinstance(chirp_channel, list):
            chirp_channel = chirp_channel[0]
        index_chirp = channels.index(chirp_channel) # index of the chirp channel
        sig_chirp = sig[:, channels[index_chirp]] # emitted chirp
        channels_sig = [c for c in channels if c != chirp_channel] # channels we want to decorrelate from chirp signal
        nb_channels = len(channels_sig) # number of channels
        
        sig_plate = np.zeros([len(sig[:, 0]), nb_channels], dtype=float) # signals to decorrelate from chirp
        for c in range(nb_channels):
            sig_plate[:, c] = sig[:, channels_sig[c]] # channels to decorrelate from chirp
            

        ## find chirp start and end indexes in sig_chirp 
        diff_phase = abs(np.diff(np.real(np.exp(1j*np.angle(sig_chirp))))) # compute the phase of the chirp signal
        matches = [x for x in range(len(diff_phase)) if diff_phase[x] > 0.5] # find all phase changes
        
        # Find start and stop indexes of chirps (detect unusually long time period between phase changes => new chirp)
        L = (1/(2*f1))*fs # number of indexes for one half period of the smallest frequency of the chirp
        index_start = []
        index_stop = []
        ii = 0
        while ii < len(matches) - 1:
            if matches[ii+1] - matches[ii] > 0.7*L: # if the number of indexes between two phase changes is > 0.9L, a new chirp begins
                index_start.append(matches[ii+1]) # store the start index of the new chirp
                index_stop.append(matches[ii]) # store the end index of the last chirp
            ii += 1
        
        # clean multi detections of phase changes
        clean_ind_1 = [index_start[0]]
        for i in index_start[1:]:
            if i - clean_ind_1[-1] > 0.5*duration*fs: # remove detections if they are closer than the duration of the chirp
                clean_ind_1.append(i)
        clean_ind_2 = [index_stop[0]]
        for i in index_stop[1:]:
            if i - clean_ind_2[-1] > 0.5*duration*fs:
                clean_ind_2.append(i)
                
        try:
            index_start, index_stop = clean_ind_1[1], clean_ind_2[2] # pick indexes of the first detected full chirp
        except:
            raise Exception("Chirp not found !")
        
        # store the detected chirp and sig indexes
        mes_chirps = sig_chirp[index_start : index_stop]
        mes_sig = {}
        for c in range(nb_channels): 
            mes_sig[c] = sig_plate[index_start : index_stop, c]
            
        L_sig = int(duration*fs) # length of the returned data
        L = len(mes_chirps) # length of the chirp
        impulse = np.zeros([L_sig, nb_channels], dtype=float) # initialize vector for the impulse
        for c in range(nb_channels):
            
            # zero padding of the chirp and signals (is it useful ?)
            chirp_mes = np.zeros([L*3])
            sig_mes = np.zeros([L*3])
            chirp_mes[L : 2*L] = mes_chirps/max(mes_chirps) * signal.tukey(L, 0.05) # windowing below correlation
            sig_mes[L : 2*L] = mes_sig[c]/max(mes_sig[c]) * signal.tukey(L, 0.05)  
        
            nb = 2**(ceil(log(chirp_mes.shape[0]) / log(2))) # length of the fft (ceil(n) is the smallest integrer above n)
            ftchirp = np.fft.rfft(chirp_mes, nb) # fft(c(t)) fft of the chirp signal (np = numpy)
            ftresult = np.fft.rfft(sig_mes, nb) # ftt(r(t)) : fft of the recorded signal
            corre = np.fft.irfft(ftresult*np.conjugate(ftchirp)) # r(t) * c(-t) = h(t) * c(t) * c(-t)
            # do the correlation between the recorded signal and the conjugate of the emitted chirp on every channels
            impulse[:, c] = np.concatenate((corre[int(len(corre) / 2) : ], corre[:int(len(corre) / 2)]))[int(len(corre) / 2 * 0.99) : int(len(corre) / 2 * 0.99) + L_sig] # store the correlation for each channel in impulse
            
            impulse[:, c] = impulse[:, c] / max(impulse[:, c]) # normalize the impulse
            
        t_s = np.linspace(0, duration, L_sig) # time signal in sec
        
        return (t_s, impulse)
    
            
        
