# Pypico
Python class to communicate with Picoscope series 5000 datalogger

This folder contains the Pypico class and some example scripts to emit and record signals with a Picoscope series 5000.
Please refer to the Pypico file for comprehensive help on how to use the functions of the class.
Further explanations are given below.

To use the class in a python script, write: 
(Please also refer to examples data acquisition scripts: Simple_acquisition.py, Simple_impulse_acquisition.py and Long_acquisition_pico.py)

    from Pypico import Pypico
    p = Pypico() # create a Pypico object and starts the Picoscope.

Then, you have to initialize the channels you intend to use, set the voltage range on these channels using the set_channels function:

    p.set_channels(channels=[0, 1], channel_range=[0.2, 1]) # channels (A:0, B:1), channel_range (V)

Set the trigger after which the Picoscope will start to save data with function set_trigger:
Example:

    p.set_trigger(trig_channel=0, threshold_mV=10) # set the trigger on channel A

To simply record data on a channel, use the function simple_record:

    time_s, sig = p.simple_record([0], fs, duration) 
This function records time and signal on channel A, at sampling frequency "fs" and for a duration "duration"

To emit a chirp signal and record the impulse response of the excited structure, use function chirp_and_rec:

    time_s, sig = p.chirp_and_rec([0, 1], [1], fs, duration, f1, f2)
This function records time and impulse response on channel A, at sampling frequency "fs" and for a duration "duration"
after the emission of a chirp signal between frequency f1 and f2 (recorded on channel B)


Attention: Practically, it is advised to write your code in a "try:, finally:" structure by finishing with p.stop_pico()
Doing so, the Picoscope will be shut down even if an exception raises (even with keyboard interrupt).
If the Picoscope is not closed properly, it may crash and you will have to restart the kernel of your python IDE.
Example:

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
    
    
An example of script to make successive measurements is given in Long_acqusition_pico.py and the script to plot the data is Plot_decorrelation_data.py
Some utility functions for signal processing are included in Func_lib_corrosion.py
