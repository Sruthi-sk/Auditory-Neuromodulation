#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 10:36:26 2022

@author: sruthisk


Stimuli related effect 
- power during stim, increase in variation - bar graph/ box plot
Comparison of mean power across 2s window
PAC - max (custom codes / python / ccs_toolbox)


DOUBTS
- notch filter, filter for 0.5 to 40? no
- get theta powers before or after epoching into 2s blocks? - before

- relative bandpowers - entire data - I can get relative bp if I dont filter - no relative - wont see the change
- men or yasa's algo on filtered data - they are taking psd' - welch not multitaper

- Per Individual - box plot of theta powers - tone vs white noise s pink noise vs music ?
- 4 files for 2s variation ?


ALGO ---------------------------------
# separate csv for bandpowers of all subjects - 4 files for bandpower, 4 files for PAC
# (Subname, StimType, PreStim2s, DurStimFirstHalf, DurStimSecondHalf, PostStim2s ) 

Matlab: theta filter - epoch of 8s - remove baseline - empty array of 25 vals - compute prestim,stim1,stim2,poststim values

# for all freq bands or only theta
    # for all subjects
        #eeg data - choose channel , filter 0.5 and 25, notch filter, 
        #for marker in all_stimuli (1-4)
            #epoch -2 to 6 around stimuli
            remove artifacts? - 
            # remove baseline of -2 to 0 - subtract mean and divide by sd - we are filtering by theta then removing baseline
            #  timelock to 25 trials
            # welch power for prestim, durstim1 (epoch.times from times 0-2), durstim2, poststim
            # calculate mean and sd 
            # calculate pac

"""
#%%

import time
# import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import mne
import yasa

from sklearn.ensemble import IsolationForest


# def find_nearest(array, value):
#     array = np.asarray(array)
#     idx = (np.abs(array - value)).argmin()
#     return array[idx]

#%%
paths = r'/serverdata/ccshome/sruthisk/NAS/DST_BDTD_2021/IITGN_codes/new_recordings/'
subjects = os.listdir(paths)

os.chdir(r'/serverdata/ccshome/sruthisk/NAS/DST_BDTD_2021/IITGN_codes/files_for_stats')

target_channel_dict = {}
# nontarget_channel_dict = {}
for sub in subjects:
    name = os.path.basename(os.path.normpath(sub))
    channel =  int(name[-1]) 
    target_channel_dict[name] = channel
    # nontarget_channel_dict[name] = '3' if channel == 2 else '2'

sf=256

stimuli_markers = [111000,111001,111002,111003]
stimuli_names = { 111000:'tone440hz',111001:'whitenoise', 111002:'pinknoise' , 111003:'music'}

freq_bands = {'Delta':[1,4],'Theta':[4,8],'Alpha':[8,12],'Beta':[13,30]}


#%% Bandpowers - 1 files

block_time = int(2*sf)

    
df_bandpowers_allsubjects = pd.DataFrame(None)

for subname in subjects:
    
    # subname= subjects[2]
    print('\n ######################',subname)
    
    sub = paths + subname #subjects[5]
    # name = os.path.basename(os.path.normpath(sub))
    # print(name)
    channel = target_channel_dict[subname]
    sub_csv_file = [f for f in os.listdir(sub) if '.csv' in f][0]
    
    raw_data = pd.read_csv(sub+'/'+sub_csv_file)
    eeg_data = raw_data.iloc[:,2:6]
    channel_data = eeg_data[str(channel)]
    
    #filter for notch, 0.5 to 40?
    chandata_for_mne = channel_data.to_numpy()
    # mne notch or DataFilter.perform_bandstop(brainflow_chunk4, sampling_rate, 48.0, 52.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    channel_data_filtered = mne.filter.filter_data(chandata_for_mne,sf,0.5,40,method='fir',fir_design='firwin')
    
    marker_data = raw_data.iloc[:,-1].to_numpy()
    # marker_idx = marker_data.nonzero() # where its not 0
    # only_markers = marker_data[marker_idx]
    marker_stimuli_idx = [i for i,mark in enumerate(marker_data) if mark in stimuli_markers]
    n_markers = len(marker_stimuli_idx)
    # stimuli_all = marker_data[marker_stimuli_idx]
    
    df_allstims_1subject = pd.DataFrame(None)
    
    for stimtype in stimuli_markers:
    
        stimname = stimuli_names[stimtype]
        print(stimname)
        
        #25 trials
        current_stim_idx = [i for i,mark in enumerate(marker_data) if mark == stimtype]
        n_stimtype = len(current_stim_idx)
        #epoch around stimuli (-2 to 6)
        current_stim_epochs = [ channel_data_filtered[ stim - block_time : stim + 3*block_time ] for stim in current_stim_idx ]
        # remove epoch with artifacts - isolation forest?
        
        current_stim_npepochs = np.array(current_stim_epochs)
        
        ilf = IsolationForest(contamination='auto', max_samples='auto', verbose=0, random_state=42)
        good = ilf.fit_predict(current_stim_npepochs)
        print("no of bad epochs = ",(good==-1).sum())
        good_stim_epochs = current_stim_npepochs[good==1]   
        # bad_stim_epochs = current_stim_npepochs[good==-1]
        
        df_allbands = pd.DataFrame(None)
        for band,bvalues in freq_bands.items():
            print(band)
            low,high = bvalues[0],bvalues[1]
            freqs=np.arange(low,high,step=0.5)                                                                                            
            ncycles=3 
            
            raw_data = good_stim_epochs.reshape(len(good_stim_epochs),1,-1) #reshape as (epochs,channels, timepoints)
            power = mne.time_frequency.tfr_array_morlet(raw_data,sf,freqs,ncycles,use_fft=True,verbose=True,output='power')[:,0,:,:]  #single trial power:: helps provide additional information that is obscured or simply not available in the average responses.
            power_band = power.mean(axis=1)
            
            #remove filtered data's baseline of prestim - subtract mean and divide by sd
            inst_power_prestim = power_band[ : , : block_time ]            
            baseline_mean = np.mean(inst_power_prestim,axis=1, keepdims=True)
            baseline_sd = np.std(inst_power_prestim,axis=1, keepdims=True)
            
            power_band_normalized = (power_band - baseline_mean)/baseline_sd
            
            # each epoch - split into 4 blocks
            power_prestim = power_band_normalized[ : , : block_time ].mean(axis=1)
            power_stim1 = power_band_normalized[ : , block_time : int(2*block_time) ].mean(axis=1)
            power_stim2 = power_band_normalized[ : , int(2*block_time) : int(3*block_time) ].mean(axis=1)
            power_poststim = power_band_normalized[ : , int(3*block_time) :  ].mean(axis=1)
            
            df_band = pd.DataFrame(None)
            df_prestim = pd.DataFrame(power_prestim,columns=[band]) ; df_prestim['Time'] = 'prestim'
            df_stim1 = pd.DataFrame(power_stim1,columns=[band]); df_stim1['Time'] = 'stim1'
            df_stim2 = pd.DataFrame(power_stim2,columns=[band]); df_stim2['Time'] = 'stim2'
            df_poststim = pd.DataFrame(power_poststim,columns=[band]); df_poststim['Time'] = 'poststim'
            
            df_band = pd.concat([df_band,df_prestim,df_stim1,df_stim2,df_poststim])
            
            df_allbands = pd.concat([df_allbands,df_band],axis=1)
            
        df_allbands = df_allbands.loc[:,~df_allbands.columns.duplicated()].copy()
        df_allbands['StimType'] = stimname
        #append to df of all stims per subject
        df_allstims_1subject = pd.concat([ df_allstims_1subject, df_allbands], axis=0)  
        
    #add subject name as column 
    df_allstims_1subject['name'] = subname
    
    #APPEND TO DF OF ALL SUBJECTS PER BAND
    df_bandpowers_allsubjects = pd.concat([ df_bandpowers_allsubjects, df_allstims_1subject], axis=0)  
        
    
filename = 'AuditoryNM_allStimTypes_BandpowersNormalized.csv'
df_bandpowers_allsubjects.to_csv(filename)
        
            
#%% mean and sd  plot of bandpowers (from csv file)

import pandas

df_bandpowers_allsubjects = pd.read_csv(filename,index_col=0)
#2104 data
# if all 600 (25*4types* 6 stimuli) epochs were good, then 600 * 4 - prestim, stim1,stim2,poststim

    
    
    
    