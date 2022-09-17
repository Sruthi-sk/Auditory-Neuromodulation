# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 12:15:16 2022

@author: ws4
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

print('Check which eeg channel is best !')

#%% Connect to brainflow

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

print('\nPlotting real time data')
time.sleep(2)
# data = board.get_current_board_data (256) # get latest 256 samples or less, doesnt remove them from internal buffer
data = board.get_board_data()  # get all data and removes it from internal buffer

# board.stop_stream()
# board.release_session()


#%% plot the 4 eeg channels - verify it is good eeg - or plot real time eeg 
# data = board.get_board_data() #clear buffer
# fig, axs = plt.subplots(len(eeg_channels))
# time.sleep(3)
# # dataz = board.get_current_board_data(sf*3)
# dataz = board.get_board_data()

# for i in range(len(eeg_channels)):
    # channel = i+1
    # DataFilter.detrend(dataz[channel], DetrendOperations.CONSTANT.value) # DetrendOperations.LINEAR.value
    # DataFilter.perform_bandpass(dataz[channel], sampling_rate, 15.0, 28.0, 2,FilterTypes.BUTTERWORTH.value, 0) # filter (1,29)
    # DataFilter.perform_bandstop(dataz[channel], sampling_rate, 50.0, 4.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    # axs[i].plot(dataz[time_channel],dataz[i+1]) # dataz[14] - time channel
# plt.show()

#%% See real time all channels - verify it is good eeg 

# mini_sample_size = int(sf/8)
fig, axs = plt.subplots(len(eeg_channels),figsize=(20,10))
count = 0
wait_max=10
start_time, current_time = time.time(), time.time()
while time.time() < (start_time + wait_max):
    axs[0].cla()
    axs[1].cla()
    axs[2].cla()
    axs[3].cla()
    newest_data = board.get_current_board_data(2*sf)
    count+=1
    # DataFilter.detrend(newest_data, DetrendOperations.CONSTANT.value)
    DataFilter.perform_bandpass(newest_data[1], sampling_rate, 15.0, 29.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandpass(newest_data[2], sampling_rate, 15.0, 29.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandpass(newest_data[3], sampling_rate, 15.0, 29.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandpass(newest_data[4], sampling_rate, 15.0, 29.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    axs[0].plot(newest_data[1])  #newest_data[time_channel],
    axs[1].plot(newest_data[2])
    axs[2].plot(newest_data[3])
    axs[3].plot(newest_data[4])
    plt.draw()
    plt.pause(0.01)

#%%

print('Check which eeg channel is best !')


