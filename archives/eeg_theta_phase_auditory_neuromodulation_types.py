# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:34:18 2022

@author: Sruthi Kuriakose
"""

"""
From commandline,
python eeg_theta_phase_auditory_neuromodulation_types.py.py --channel=2

Algorithm

Get eeg from simulated data / from Muse
- pause time of 0.01 if no of samples we get from buffer every call is low - unnecessary to pause but just for plotting
- get real time eeg data of n samples
- of last 10? samples, find phases - - Maybe if no pause - get every single sample and check phase
Send audio stimuli at phase 0 
Add marker when stimuli is sent
Save eeg file with marker

paradigm: 
200 stimulations with 4s period of stim 
presented in trains of five (types? tones) 
with interstimuli interval of 4-6s and 
20 stimuli per type per subject


Useful links
https://www.pygame.org/docs/ref/mixer.html
https://www.pygame.org/docs/ref/music.html#pygame.mixer.music.load 

Check time as (2s prestim + 4s stim + 2s poststim + 3s interstim) * 2 trials * 3 audio files 
Check where file gets saved


Change because this code uses the old brainflow API for bandpass filter- center freq, bandwidth
"""

#%% import libs

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds #LogLevels,
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations # AggOperations, WindowFunctions, 

import time

# import os
import matplotlib.pyplot as plt
import numpy as np
import pygame
# import winsound
import scipy.signal as signal
import pandas as pd
import joblib
import argparse

#%% Connect to brainflow
BoardShim.enable_dev_board_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--channel', type=int, help='select eeg channel', required=True) #, default=3
args = parser.parse_args()
channel = args.channel
print('CHANNEL CHOSEN IS ',channel)
    
params = BrainFlowInputParams()
# board_id = BoardIds.SYNTHETIC_BOARD.value
board_id = BoardIds.MUSE_S_BOARD.value
sf= sampling_rate = BoardShim.get_sampling_rate(board_id)
board_descr = BoardShim.get_board_descr(board_id)
time_channel = BoardShim.get_timestamp_channel(board_id)
eeg_channels = BoardShim.get_eeg_channels(board_id)
board = BoardShim(board_id, params)

board.prepare_session()

board.start_stream()

time.sleep(2)
data = board.get_current_board_data (256) # get latest 256 samples or less, doesnt remove them from internal buffer
# data = board.get_board_data()  # get all data and remove it from internal buffer

# board.stop_stream()
# board.release_session()


#%% load sound - continuous tone of diff frequencies, music, pulses 

#continuous tone
# pygame.mixer.init()
# pygame.mixer.music.load(r"440Hz.wav") # 30s file
# pygame.mixer.music.set_volume(0)

# pygame.mixer.music.unload()
# pygame.mixer.music.stop()  
# pygame.mixer.music.pause() 

# win_data = window_data

def findphaseidx(win_data):
    # Find all phases in window_data
    win_data = win_data*1e6
    analytic_signal = signal.hilbert(win_data) # converts to complex signal
    #The instantaneous phase corresponds to the phase angle of the analytic signal.  
    # instantaneous frequency can be obtained by differentiating the instantaneous phase in respect to time
    phase_all = np.angle(analytic_signal)
    # plt.plot(win_data)
    # plt.plot(phase_all)
    phase1 = np.round_(phase_all,1)
    
    #checking when phase=0 by checking when it moves from -ve to +ve
    array_sign=np.sign(phase1)
    array_sign[np.where(array_sign==0)]=1
    bool_phase0_idx = (np.diff(array_sign) ==2)   #!=0 for crest and trough  # diff = 2 when phase shifting from -ve to +ve
    
    # idx_0ph = np.asarray(np.where(bool_phase0_idx ==1))+1
    
    return bool_phase0_idx 



#%% Paradigm parameters
# 5s prestim, 5s stim, 5s poststim - repeat 5x times for each type of sound
# which channel for phase 
# channel=3

pygame.mixer.init()
audio_files = [ r"sounds\440Hz.wav",
               # r"sound\white_noise.mp3",
                r"sounds\pink_noise.mp3",
               r"sounds\al-andalus.mp3",
    ]

# Declare window size 
mini_window = 15  # window for checking if phase=0
pause_time = mini_window/sf ## optimal pause_time to collect that many samples
window_size = 2*sf # longer window for filtering etc

# time for stimulation
time_stim = 4  # insert marker 555 before stim and after
time_pre = 2  # insert marker 111 before pre
time_post = 2 # insert marker 333 after post
time_inter_stim = 3
n_trials = 2  

count = 0 # to count no of times we call from buffer - atleast (sf/mini_window) times per sec to avoid missing data 
        # - that many unique non-overlapping segments 
phase_markers = 0
sound_markers = []  #insert marker 999 to indicate sound is produced
count_no_0phase = 0

# pygame.mixer.music.play(loops=-1) #  repeats indefinitely
# time.sleep(1)

time_delay_max = (1/9) - (mini_window/sf)# time allowed to sleep after detecting phase 0
time_delay_peak = (1/9) - (mini_window/sf)
# 1 second - max 9 as theta freq
# between two peaks, atleast 100ms
#%%
_=board.get_board_data() # to clear buffer
sample_start = board.get_board_data_count()

# expectd no of markers = no of trials * (start 111+ stop 333 + 2*555)+ no of soundmarkers in total

start_trial = time.time()


for repeat in range(1):
    
    for ix,audio in enumerate(audio_files):
        pygame.mixer.music.load(audio) # 30s file
        print(audio)
        pygame.mixer.music.set_volume(0)
        pygame.mixer.music.play(loops=-1) #  repeats indefinitely   #loops=-1     
        
        for i in range(n_trials):  
            
            board.insert_marker(111000+ix) # starting prestim baseline
            time.sleep(time_pre)
            print('Trial ',i+1)
           
            #stimulation
            start_stim_time, current_stim_time = time.time(), time.time()
            board.insert_marker(555 ) #indicate stim
            detected = False
            while current_stim_time < (start_stim_time + time_stim):
                
                # plt.cla()
                pygame.mixer.music.set_volume(0.4)
                
                if detected == True:
                    time.sleep(time_delay_max)
                detected = False
                
                window_data = board.get_current_board_data(window_size)[channel]
                count+=1
                DataFilter.detrend(window_data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(window_data, sampling_rate, 6.0, 4.0, 2,FilterTypes.BUTTERWORTH.value, 0)  
                DataFilter.perform_bandstop(window_data, sampling_rate, 50.0, 4.0, 2,FilterTypes.BUTTERWORTH.value, 0)
        
                # produce sound if phase 0 present in mini_window
                phase0_all = findphaseidx(window_data)
                idx_0ph = np.asarray(np.where(phase0_all ==1))+1
                phase_curr_idx = idx_0ph[(idx_0ph > window_data.shape[0]-mini_window)]
                
                phase_curr_bool = phase0_all[-mini_window:] # if bools instad of index
                n_0phase = phase_curr_bool.sum()
                
                # find index where phase_curr_bool == True
                # idx_phase = np.where(phase_curr_bool == True)[0][0]
                
                try: # if n_0phase>0: #if phase_curr.shape[0]!=0:
                    idx_phase = phase_curr_bool.tolist().index(1)
                    phase_markers+=1
                    # print('phase 0 detected')
                    pygame.mixer.music.set_volume(0.6)
                    sound_markers.append(board.get_board_data_count())  #no of samples
                    board.insert_marker(999)
                    detected = True
                    time_delay_max = time_delay_peak + (idx_phase/sf) #(1/9)- (mini_window/sf)
                except ValueError: #else:
                    count_no_0phase +=1
                
                # time.sleep(0.01) # try removing - more count
            
                #Plotting
                # plt.plot(window_data)
                # plt.scatter(idx_0ph,window_data[idx_0ph],color="red")
                # plt.draw()
                # plt.axvline(x=window_data.shape[0]-mini_window,color='r')
                # plt.pause(0.02) #pause_time
                
                time.sleep(0.02)
            
                current_stim_time=time.time() 
            
            pygame.mixer.music.set_volume(0.0)
            print('stim over')
            board.insert_marker(555) #indicate stim over
            
            time.sleep(time_post)
            board.insert_marker(333) #indicate poststim period over
            
            time.sleep(time_inter_stim) # Inter-stimuli interval

        pygame.mixer.music.unload()

    
pygame.mixer.music.stop()
end_trial=time.time()
trial_time_taken = np.round(end_trial-start_trial,3)
print(" total paradigm time ",trial_time_taken)
print(' On PC, measured time was 66.09s  ; (2s prestim + 4s stim + 2s poststim + 3s interstim) * 2 trials * 3 audio files ')

# print(" no of buffer calls ",count)   
# print('per second, no of buffer calls should be atleast (if the delay after detecting phase not there): ', int(sf/mini_window),
#       ' and we have :',int(count/time_stim), ' calls per sec')
# print(" no of calls with no sound ",count_no_0phase)
# print(" no of phase zero markers ",phase_markers,len(sound_markers))
# print(" max no of phase zero markers assuming theta=8, 8*stim_time : ",8*time_stim)

# print('max freq of person ',phase_markers/time_stim,'??')

#%%

sample_end = board.get_board_data_count()
print(sample_end)  
# data_whole_rec =  board.get_current_board_data(sample_end-sample_start) #.reshape(-1).astype('float64') 
# will only work in above line is called immediately - should miss samples - better to take whole data
# cross check buffer size

recorded_data  = board.get_board_data()
df = pd.DataFrame(np.transpose(recorded_data))

df.iloc[sound_markers,:] #- verify that markers have been recorded in marker_ch

paths = r"eeg_muse_test/"
fname = 'complete_paradigm_060922_'

# fname = 'piano_paradigm_270822'

df.to_csv(paths+'eeg_'+fname+'.csv')

# soundmarkers_ecg = [np.array(sm)-sample_start for sm in soundmarkers]  # only if saved data is using get_current_board_data
print("sound markers : ",sound_markers) #_ecg
joblib.dump(sound_markers,paths+'soundmarkers_'+fname+'.pkl')

# r=joblib.load('eeg_muse_test/'+'responses.pkl')


#%% End buffer
board.stop_stream()
board.release_session()

#%%


