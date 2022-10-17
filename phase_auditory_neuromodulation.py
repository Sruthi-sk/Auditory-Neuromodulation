# -*- coding: utf-8 -*-
"""
Created on Tue Aug 23 15:34:18 2022
"""

"""
Neuromodulation Paradigm
freq = theta 

Algorithm

Get eeg from simulated data / from Muse
- pause time of 0.01 if no of samples we get from buffer every call is low - unnecessary to pause but just for plotting
- get real time eeg data of n samples
- of last 10? samples, find phases - inherent buffer 10-12
Send audio stimuli at phase 0 
Add marker when stimuli is sent
Save eeg file with marker

paradigm: 
    #----- trains of 5 
        # different auditory stimuli types :
        # (auditory pulses \\ amplitude modulation of pure tones \\ 
        # amplitude modulation of white noise, amplitude modulation of pink noise \\ 
        # amplitude modulation of music clip)
#----- 100 stim per subject, 20 stimuli per type per subject
#----- stim period 4s , interstimuli interval of 4-6s

Check time as (2s prestim + 4s stim + 2s poststim + 3s interstim) * 2 trials * 3 audio files 
Check where file gets saved
"""

#%% import libs

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds #LogLevels,
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations # AggOperations, WindowFunctions, 

import time

import numpy as np
import scipy.signal as signal
import pandas as pd
import pygame
import pyedflib
import joblib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import argparse
# import winsound

import os
os.chdir('C:/Users/ws4/Documents/Sruthi_JRF/neuromodulation/IITG/')


#%% Connect to brainflow

def connect_board():
    
    BoardShim.enable_dev_board_logger()
    
    params = BrainFlowInputParams()
    # board_id = BoardIds.SYNTHETIC_BOARD.value
    board_id = BoardIds.MUSE_S_BOARD.value
    sf= sampling_rate = BoardShim.get_sampling_rate(board_id)
    # board_descr = BoardShim.get_board_descr(board_id)
    time_channel = BoardShim.get_timestamp_channel(board_id)
    eeg_channels = BoardShim.get_eeg_channels(board_id)
    board = BoardShim(board_id, params)
    
    board.prepare_session()
    
    board.start_stream()
    
    time.sleep(2)
    # data = board.get_current_board_data (256) # get latest 256 samples or less, doesnt remove them from internal buffer
    # data = board.get_board_data()  # get all data and remove it from internal buffer
    return board,sf,eeg_channels,time_channel

# board.stop_stream()
# board.release_session()

board,sampling_rate,eeg_channels,time_channel = connect_board()
sf=sampling_rate
#%% #%% plot the 4 eeg channels - verify it is good eeg - or plot real time eeg 

data = board.get_board_data() #clear buffer
fig, axs = plt.subplots(len(eeg_channels))
time.sleep(3)
# dataz = board.get_current_board_data(sf*3)
dataz = board.get_board_data()

for i in range(4):
    channel = i+1
    DataFilter.detrend(dataz[channel], DetrendOperations.CONSTANT.value) # DetrendOperations.LINEAR.value
    DataFilter.perform_bandpass(dataz[channel], sampling_rate, 15.0, 28.0, 2,FilterTypes.BUTTERWORTH.value, 0) # filter (1,29)
    # DataFilter.perform_bandstop(dataz[channel], sampling_rate, 50.0, 4.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    axs[i].plot(dataz[time_channel],dataz[i+1]) # dataz[14] - time channel
plt.show()


#%% See real time all channels -----------------------------------------------------------

# mini_sample_size = int(sf/8)
fig, axs = plt.subplots(len(eeg_channels),figsize=(10,10))
count = 0
wait_max=10
start_time, current_time = time.time(), time.time()
while time.time() < (start_time + wait_max):
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    axs[3].cla()
    newest_data = board.get_current_board_data(2*sampling_rate)
    count+=1
    # DataFilter.detrend(newest_data, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(newest_data[1], sampling_rate, 0.5, 45, 1 ,FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandpass(newest_data[2], sampling_rate, 0.5, 45, 1 ,FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandpass(newest_data[3], sampling_rate, 0.5, 45, 1 ,FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandpass(newest_data[4], sampling_rate, 0.5, 45, 1 ,FilterTypes.BUTTERWORTH.value, 0)
    axs[0].plot(newest_data[1])
    axs[1].plot(newest_data[2])
    axs[2].plot(newest_data[3])
    axs[3].plot(newest_data[4])
    plt.draw()
    plt.pause(0.5)
    

#%% Visualize continuous real time data without appending it

channel=3
# mini_sample_size = int(sf/8)
mini_window = 15   # when 10, missed rpeaks for peak detection

plt.figure(figsize=(10,5))
count = 0
wait_max=10
start_time, current_time = time.time(), time.time()
while time.time() < (start_time + wait_max):
    plt.cla()
    newest_data = board.get_current_board_data(3*sampling_rate)[channel]
    count+=1
    DataFilter.detrend(newest_data, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(newest_data, sampling_rate, 0.5, 45, 1 ,FilterTypes.BUTTERWORTH.value, 0)
    
    # # time.sleep(0.001)
    # plt.plot(times,data_2s)
    plt.plot(newest_data)
    plt.draw()
    plt.axvline(x=newest_data.shape[0]-mini_window,color='r')   # plt.ylim(-50,50)
    plt.pause(0.5)

end=time.time()
print("totaltime:",end-start_time)
print("sample_size : ",mini_window)
print("no of buffer calls",count)
print(wait_max*sampling_rate,"<", mini_window*count,"? if yes, samples should not be missing")
print('per second, count should be atleast: ', int(sampling_rate/mini_window),' and we have :',int(count/wait_max), ' calls per sec')

    
#%% finding phase

# win_data = window_data
def findphaseidx(win_data):
    # Find all phases in window_data
    win_data = win_data #*1e-6
    analytic_signal = signal.hilbert(win_data) # converts to complex signal
    #The instantaneous phase corresponds to the phase angle of the analytic signal.  
    # instantaneous frequency can be obtained by differentiating the instantaneous phase in respect to time
    phase_all = np.angle(analytic_signal)
    # plt.plot(win_data)
    # plt.plot(phase_all)
    phase_all_round = np.round_(phase_all,1)
    
    #checking when phase=0 by checking when it moves from -ve to +ve
    array_sign=np.sign(phase_all_round)
    array_sign[np.where(array_sign==0)]=1
    bool_phase0_idx = (np.diff(array_sign) ==2)   #!=0 for crest and trough  
    # diff = 2 when phase shifting from -ve to +ve
    
    # idx_0ph = np.asarray(np.where(bool_phase0_idx ==1))+1
    return bool_phase0_idx 

#%% load sound & Paradigm parameters -----------------------------------------------------------
# 2s prestim, 4s stim, 2s poststim , 3s interstim- repeat 5x times for each type of sound
# which channel for phase 
pygame.mixer.init()
audio_files = [ r"sounds\440Hz.wav", r"sounds\white_noise.mp3",
                r"sounds\pink_noise.mp3", r"sounds\al-andalus.mp3" ] 
pygame.mixer.music.set_volume(0)
# pygame.mixer.music.play()   .unload()  .stop()  .pause() 
# pygame.mixer.music.play(loops=-1) #  repeats indefinitely


# Declare window size ------------------------------------------------------------
mini_window = 15  # window for checking if phase=0
pause_time = mini_window/sf ## optimal pause_time to collect that many samples
window_size = 2*sf # longer window for filtering etc

# time for stimulation
time_stim = 4  # insert marker 555 before stim and after
time_pre = 2  # insert marker 111 before pre
time_post = 2 # insert marker 333 after post
time_inter_stim = 3
n_trains = 1
n_trials = 2 # in 1 train of same audio file

count = 0 # to count no of times we call from buffer - atleast (sf/mini_window) times per sec to avoid missing data 
        # - that many unique non-overlapping segments 
phase_markers = 0
sound_markers = []  #insert marker 999 to indicate sound is produced
count_no_0phase = 0


time_delay_max = (1/9) - (mini_window/sf)# time allowed to sleep after detecting phase 0
time_delay_peak = (1/9) - (mini_window/sf)
# 1 second - max 9 as theta freq
# between two peaks, atleast 100ms

#%% Actual paradigm

_=board.get_board_data() # to clear buffer
sample_start = board.get_board_data_count()
# expectd no of markers = ntrains*n_sounds*no of trials * (start 111+ stop 333 + 2*555)+ no of soundmarkers in total

start_trial = time.time()


for repeat in range(n_trains):
    
    for ix,audio in enumerate(audio_files):
        pygame.mixer.music.load(audio) # 30s file
        print(audio)
        pygame.mixer.music.set_volume(0)
        pygame.mixer.music.play(loops=-1) #  repeats indefinitely   #loops=-1     
        
        for i in range(n_trials):  
            
            board.insert_marker(111000+ix) # starting prestim baseline
            print('Trial ',i+1)
            time.sleep(time_pre)
           
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
                
                orig_window_data = board.get_current_board_data(window_size)[channel]
                window_data = orig_window_data.copy()
                count+=1
                DataFilter.detrend(window_data, DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(window_data, sampling_rate, 4.0, 8.0, 1,FilterTypes.BUTTERWORTH.value, 0)
                # DataFilter.perform_bandstop(window_data, sampling_rate, 50.0, 4.0, 2,FilterTypes.BUTTERWORTH.value, 0)
                # plt.plot(window_data)
                
                # produce sound if phase 0 present in mini_window
                phase0_all = findphaseidx(window_data)
                # DOUBT - SHOULDN'T THERE BE ONLY MAX OF 16 true values ( 0 PHASES) IN THE 2s DATA
                idx_0ph = np.asarray(np.where(phase0_all ==1))+1
                # phase_curr_idx = idx_0ph[(idx_0ph > window_data.shape[0]-mini_window)]
                
                phase_curr_bool = phase0_all[-mini_window:] # if bools instad of index # if any phase of mini window is 0
                # n_0phase = phase_curr_bool.sum()
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
                
                #Plotting
                # plt.plot(window_data)
                # plt.scatter(idx_0ph,window_data[idx_0ph],color="red")
                # plt.draw()
                # plt.axvline(x=window_data.shape[0]-mini_window,color='r')
                # plt.pause(0.02) #pause_time
                
                time.sleep(0.02) ## try removing - more count
            
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
# currently (2s prestim + 4s stim + 2s poststim + 3s interstim) * 2 trials * 4 audio files * 1 train = 11*8=88s


print(" no of buffer calls ",count) 
# print(" no of buffer calls with no sound ",count_no_0phase)  
# print('per second, no of buffer calls should be atleast (if the delay after detecting phase not there): ', int(sf/mini_window),
#       ' and we have :',int(count/time_stim), ' calls per sec')
print(" no of phase zero markers ",phase_markers,len(sound_markers))
print(" max no of phase zero markers assuming theta=8, 8*stim_time : ",8*time_stim)
print('max freq of person ',phase_markers/time_stim,'??')


#%%

sample_end = board.get_board_data_count()
print(sample_end)  
# data_whole_rec =  board.get_current_board_data(sample_end-sample_start) #.reshape(-1).astype('float64') 
# will only work in above line is called immediately - should miss samples - better to take whole data
# cross check buffer size to see max time we can store data - 60k samples?


recorded_data  = board.get_board_data()
df = pd.DataFrame(np.transpose(recorded_data))

df.iloc[sound_markers,:] #- verify that markers have been recorded in marker_ch
#%%
import datetime
paths = r"eeg_nm_data/"
fname = 'complete_paradigm_aiswarya_'+str(datetime.date.today())

# fname = 'piano_paradigm_270822'

df.to_csv(paths+'eeg_'+fname+'.csv')

# soundmarkers_ecg = [np.array(sm)-sample_start for sm in soundmarkers]  # only if saved data is using get_current_board_data
print("sound markers : ",sound_markers) #_ecg
soundfile = paths+'soundmarkers_'+fname+'.pkl'
joblib.dump(sound_markers,soundfile)

# r=joblib.load('D:/CCS_Users/sruthi/ecg/'+'responses.pkl')


#%%

edf_file = paths+'eeg_'+fname+'.edf' 
# pyedflib.highlevel.write_edf_quick(edf_file, signals=recorded_data[eeg_channels], sfreq=sf) #without creating headers
#OR
device = board.get_device_name(board.board_id)
eeg_channels = board.get_eeg_channels(board.board_id)
signals = recorded_data[eeg_channels]
eeg_channel_names = board.get_eeg_names(board.board_id)  #ch_names = BoardShim.get_eeg_names(board.board_id)
pmin, pmax = signals.min(), signals.max()
# main_header = pyedflib.highlevel.make_header(technician='pyedflib-sk',equipment = device,
#                      patientname='patient1',recording_additional='comments',admincode='clinician1')
# signal_headers = pyedflib.highlevel.make_signal_headers(eeg_channel_names, sample_frequency=sf,
#                                          physical_min=pmin, physical_max=pmax) # dimension='uV',
# # pyedflib.highlevel.write_edf_quick(edf_file, signals=recorded_data[eeg_channels], sfreq=sf) #without creating headers
# pyedflib.highlevel.write_edf(edf_file, signals, signal_headers, main_header, digital=False)

    
#edf_file,soundfile  


#%% edf
''' 
While saving edf file
#C:/Users\domainInt1\.conda\envs\spyder-cf-52\lib\site-packages\pyedflib\edfwriter.py:99: 
#UserWarning: Physical minimum for channel 15 (CH_15) is 1019.4206211152036, which has 18 chars, 
#however, EDF+ can only save 8 chars, will be truncated to 1019.420, some loss of precision is to be expected.
'''

#%% configure neuromodulation protocol - freq time




#%% End buffer
# r=joblib.load('eeg_muse_test/'+'soundmarkers.pkl')
'''
board.stop_stream()
board.release_session()
'''
