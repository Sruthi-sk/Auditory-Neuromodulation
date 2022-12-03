# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:51:59 2022

@author: ws4
"""
'''


Average phase offset
2. Phase detection and present stimuli
- simple code - every chnk, we detect and present
Compute average phase offset for theta - check Consistency of nearby frequencies
Mne Filter/ Brainflow filter -> hilbert - phase values of each sample corresponding to marker 999 
- histogram - how it is distributed - phase difference of 36 degree
Order of filter higher is better for hilbert phase detection =- otherwise inconsistencies
 - still normal distribution so decent perf. 
Reasons for inaccuracy - (1) filter (2) hardware streaming slow - we optimized and achieved closed loop
 - why are phases not consistent?
Hardware delays - no soln - we’re not relyin on PLL
Phase detection accurate but sound presented has jitter


Methods here
1. Get markers and visualize eeg
2. Filter in theta and check if markers correspond to 0 phase visually
    Filter nearby frequencies and check if markers correspond 
3. mne filter - do hilbert and check phase values of each sample correspondinding to marker 999 
    - histogram of phase - ( compare wih brainflow filter too - maybe a reason for phase inaccuracy - realtime iir filters)
4. check how many phase 0s actually present in data and how many stimuli we presented

test eeg phase detection code
# data -> mne filter -> hilbert phase-> 
#1. check histogram or pi chart of phase of only stimuli 999
#2. check how many phase 0s missed - do after epoching stim data


'''

#%% import libs

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds #LogLevels,
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations # AggOperations, WindowFunctions, 

import time

# import os
import matplotlib.pyplot as plt
import numpy as np
# import winsound
import scipy.signal as signal
import pandas as pd
import joblib

import pyedflib
import os

import mne
import pylab

# paths=r'C:\Users\ws4\Documents\Sruthi_JRF\neuromodulation\coral neuromodulation final\subjects/'
# subjects = os.listdir(paths)
# subjects.pop(4) #remove garima's and varsha
# subjects.pop(4)
# subjects.pop(5)


#for sub in subjects:
# sub = paths + subjects[0]

paths = r'/serverdata/ccshome/sruthisk/NAS/DST_BDTD_2021/IITGN_codes/new_recordings/'

subjects = os.listdir(paths)
# subject_channel_dict = {'madhu':3,'aishwarya':2,'aishwaryapilot':3,'garima':23,'amruth':2,'adla':2}


target_channel_dict = {}
# nontarget_channel_dict = {}
for sub in subjects:
    name = os.path.basename(os.path.normpath(sub))
    channel =  int(name[-1]) 
    target_channel_dict[name] = channel
    # nontarget_channel_dict[name] = '3' if channel == 2 else '2'

sf=256

# sub_sound_file = [f for f in os.listdir(sub) if 'sound' in f][0]
# soundmarkers=joblib.load(sub+'/'+sub_sound_file)
# df.iloc[sound_markers,:]#

# stimuli_markers = [111000,111001,111002,111003]
# marker_stimuli_idx = [i for i,mark in enumerate(marker_data) if mark in stimuli_markers]
# stimuli_all = marker_data[marker_stimuli_idx]

# time1,time2 = 4*sf,6*sf
# plt.plot(np.arange(time1,time2,1),mnefiltered_data[time1:time2])
# [pylab.axvline(_x, linewidth=1, color='r') for _x in marker_stimuli_idx if _x>time1 and _x<time2]
# [pylab.axvline(_x, linewidth=1, color='g') for _x in marker_phase0_idx if _x>time1 and _x<time2]
# plt.plot(np.arange(time1,time2,1),phase_all[time1:time2])  
# plt.axhline(0,color='r')   
# plt.ylim(-10,10)

#%% presentation of visual stimulus at theta 0 phase - as per actual EEG

sub = paths + subjects[-1]
subname = os.path.basename(os.path.normpath(sub))
channel = target_channel_dict[subname]
sub_csv_file = [f for f in os.listdir(sub) if '.csv' in f][0]

raw_data = pd.read_csv(sub+'/'+sub_csv_file)
eeg_data = raw_data.iloc[:,2:6] #channels are columns 2-6
channel_data = eeg_data[str(channel)]  #.iloc[:,channel]
chandata_for_mne = channel_data.copy().to_numpy()
mnefiltered_data = mne.filter.filter_data(chandata_for_mne,sf,5,7,method='fir',fir_design='firwin')

marker_data = raw_data.iloc[:,-1].to_numpy()
marker_idx = marker_data.nonzero() # where its not 0
only_markers = marker_data[marker_idx]

stimuli_markers = [111000,111001,111002,111003]
marker_stimuli_idx = [i for i,mark in enumerate(marker_data) if mark in stimuli_markers]
# stimuli_all = marker_data[marker_stimuli_idx]
marker_phase0_idx = [i for i,mark in enumerate(marker_data) if mark==999]

plt.plot(mnefiltered_data)
[pylab.axvline(_x, linewidth=1, color='r') for _x in marker_stimuli_idx ]
[pylab.axvline(_x, linewidth=1, color='g') for _x in marker_phase0_idx ]




#%% Visualize EEG with markers

analytic_signal = signal.hilbert(mnefiltered_data) 
phase_all = np.angle(analytic_signal)
detected_phases_persound = phase_all[marker_phase0_idx]

#208272 / sf - time
s1,s2 = 208350,20860 #813*sf,816*sf 
plt.plot(np.arange(s1,s2,1),mnefiltered_data[s1:s2],label = 'eeg')
[pylab.axvline(_x, linewidth=1, color='r') for _x in marker_stimuli_idx  if _x>s1 and _x<s2]
[pylab.axvline(_x, linewidth=1, color='g') for _x in marker_phase0_idx  if _x>s1 and _x<s2]

plt.plot(np.arange(s1,s2,1),phase_all[s1:s2], label = 'hilbert phase')  
plt.axhline(0,color='r')   
plt.ylim(-10,10)
plt.legend()
plt.show()

#%% target_channel - theta

target_phase_theta = np.array([])
for subname in subjects:
    print(subname)
    
    sub = paths + subname #subjects[5]
    # name = os.path.basename(os.path.normpath(sub))
    # print(name)
    channel = target_channel_dict[subname]
    sub_csv_file = [f for f in os.listdir(sub) if '.csv' in f][0]
    
    raw_data = pd.read_csv(sub+'/'+sub_csv_file)
    eeg_data = raw_data.iloc[:,2:6]
    
    
    marker_data = raw_data.iloc[:,-1].to_numpy()
    marker_idx = marker_data.nonzero() # where its not 0
    only_markers = marker_data[marker_idx]
    
    marker_phase0_idx = [i for i,mark in enumerate(marker_data) if mark==999]
    
    channel_data = eeg_data[str(channel)]  #.iloc[:,channel]
    
    #filter nd get phase
    #detrend, bandstop? -----------------------------------------------------------------------------------------------------------------
    chandata_for_mne = channel_data.copy().to_numpy()
    mnefiltered_data = mne.filter.filter_data(chandata_for_mne,sf,5,7,method='fir',fir_design='firwin')
    analytic_signal = signal.hilbert(mnefiltered_data) # converts to complex signal
    #The instantaneous phase corresponds to the phase angle of the analytic signal.  
    # instantaneous frequency can be obtained by differentiating the instantaneous phase in respect to time
    phase_all = np.angle(analytic_signal)
    # check phase of only marker_phase0 - stimuli 999
    detected_phases_persound = phase_all[marker_phase0_idx]
    target_phase_theta = np.append(target_phase_theta,detected_phases_persound)


plt.hist(target_phase_theta )

#%%#%% target_channel - all freq bands

freq_bands = {'delta':[1,4],'theta':[5,7],'alpha':[8,12],'beta':[13,30]}
target_phase_delta, target_phase_theta, target_phase_alpha,target_phase_beta = np.array([]) , np.array([]), np.array([]), np.array([])
freq_phase_dict = {'delta':target_phase_delta, 'theta': target_phase_theta, 'alpha':target_phase_alpha, 'beta':target_phase_beta}

# for band,bvalues in freq_bands.items():
#     low,high = bvalues[0],bvalues[1]
#     print(low,high)
#     array_to_append = freq_phase_dict[band]
#     freq_phase_dict[band] = np.append(array_to_append,np.array([low,high]))
#     print(freq_phase_dict[band])


for subname in subjects:
    print('\n ####################################################',subname)
    
    sub = paths + subname #subjects[5]
    # name = os.path.basename(os.path.normpath(sub))
    # print(name)
    channel = target_channel_dict[subname]
    sub_csv_file = [f for f in os.listdir(sub) if '.csv' in f][0]
    
    raw_data = pd.read_csv(sub+'/'+sub_csv_file)
    eeg_data = raw_data.iloc[:,2:6]
    
    
    marker_data = raw_data.iloc[:,-1].to_numpy()
    marker_idx = marker_data.nonzero() # where its not 0
    only_markers = marker_data[marker_idx]
    
    marker_phase0_idx = [i for i,mark in enumerate(marker_data) if mark==999]
    
    channel_data = eeg_data[str(channel)]  #.iloc[:,channel]
    
    #filter nd get phase
    #detrend, bandstop? -----------------------------------------------------------------------------------------------------------------
    for band,bvalues in freq_bands.items():
        low,high = bvalues[0],bvalues[1]
        chandata_for_mne = channel_data.copy().to_numpy()
        mnefiltered_data = mne.filter.filter_data(chandata_for_mne,sf,low,high,method='fir',fir_design='firwin')
        analytic_signal = signal.hilbert(mnefiltered_data) # converts to complex signal
        #The instantaneous phase corresponds to the phase angle of the analytic signal.  
        # instantaneous frequency can be obtained by differentiating the instantaneous phase in respect to time
        phase_all = np.angle(analytic_signal)
        # check phase of only marker_phase0 - stimuli 999
        detected_phases_persound = phase_all[marker_phase0_idx]
        # if band=='theta':
            # plt.hist(detected_phases_persound )
            # plt.pause(1)
        array_to_append = freq_phase_dict[band]
        freq_phase_dict[band] = np.append(array_to_append,detected_phases_persound)


target_phase_theta  = freq_phase_dict['theta']
target_phase_delta = freq_phase_dict['delta']
target_phase_alpha = freq_phase_dict['alpha']
target_phase_beta = freq_phase_dict['beta']
#%%
plt.hist(target_phase_theta )
plt.hist(target_phase_delta )
plt.hist(target_phase_alpha )
plt.hist(target_phase_beta )




#%% circular hist
def circular_hist(ax, x, bins=16, density=True, offset=0, gaps=True):
    """
    https://stackoverflow.com/questions/22562364/circular-polar-histogram-in-python
    Produce a circular histogram of angles on ax.    Parameters
    ----------
    ax : matplotlib.axes._subplots.PolarAxesSubplot
        axis instance created with subplot_kw=dict(projection='polar').
    x : array
        Angles to plot, expected in units of radians.
    bins : int, optional
        Defines the number of equal-width bins in the range. The default is 16.
    density : bool, optional
        If True plot frequency proportional to area. If False plot frequency
        proportional to radius. The default is True.
    offset : float, optional
        Sets the offset for the location of the 0 direction in units of
        radians. The default is 0.
    gaps : bool, optional
        Whether to allow gaps between bins. When gaps = False the bins are
        forced to partition the entire [-pi, pi] range. The default is True.

    Returns
    -------
    n : array or list of arrays
        The number of values in each bin.
    bins : array
        The edges of the bins.
    patches : `.BarContainer` or list of a single `.Polygon`
        Container of individual artists used to create the histogram
        or list of such containers if there are multiple input datasets.
    """
    # Wrap angles to [-pi, pi)
    x = (x+np.pi) % (2*np.pi) - np.pi
    # Force bins to partition entire circle
    if not gaps:
        bins = np.linspace(-np.pi, np.pi, num=bins+1)
    # Bin data and record counts
    n, bins = np.histogram(x, bins=bins)
    # Compute width of each bin
    widths = np.diff(bins)

    # By default plot frequency proportional to area
    if density:
        # Area to assign each bin
        area = n / x.size
        # Calculate corresponding bin radius
        radius = (area/np.pi) ** .5
        colors = plt.cm.rainbow(radius )

    # Otherwise plot frequency proportional to radius
    else:
        radius = n
        colors = plt.cm.rainbow(radius/1000 )
    print(radius)
    # Plot data on ax
    ax.set_ylim((0,950))
    patches = ax.bar(bins[:-1], radius, zorder=1, align='edge', width=widths,
                      fill=True, linewidth=1,bottom=100,color=colors) #edgecolor='C0',
    # for r, bar in zip(radius, patches):
    #     bar.set_facecolor(plt.cm.rainbow(r / 1000))
    #     bar.set_alpha(0.5)
    # Set the direction of the zero angle
    ax.set_theta_offset(offset*10)
    # Remove ylabels for area plots (they are mostly obstructive)
    if density:
        ax.set_yticks([])

        
    return #n, bins, patches


#%% target electrode
plt.rcParams["figure.figsize"] = [15.00, 4.0]
fig, ax = plt.subplots(1, 4, subplot_kw=dict(projection='polar'))
# circular_hist(ax, angles, bins=32,density=True) #Visualise by area of bins
circular_hist(ax[0], target_phase_delta ,bins=32,density=False)
circular_hist(ax[1], target_phase_theta ,bins=32,density=False) #Visualise by radius of bins
circular_hist(ax[2], target_phase_alpha ,bins=32,density=False)
circular_hist(ax[3], target_phase_beta ,bins=32,density=False)
fig.suptitle('Target Electrode')
ax[0].set_title('Delta', fontstyle='italic')
ax[1].set_title('Theta', fontstyle='italic')
ax[2].set_title('Alpha', fontstyle='italic')
ax[3].set_title('Beta', fontstyle='italic')
fig.tight_layout()
plt.show()


#%%


'''
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
angles = (x+np.pi) % (2*np.pi) - np.pi
n, bins = np.histogram(x, bins=bins)
# Compute width of each bin
widths = np.diff(bins)
radii = n
ax = plt.subplot(111, projection='polar')
bars = ax.bar(bins[:-1], radii, width=widths, bottom=0.0)
for r, bar in zip(radii, bars):
    bar.set_facecolor(plt.cm.rainbow(r / 1000))
    bar.set_alpha(0.5)
plt.show()
ax.get_yticks()
ax.set_yticks([])
'''



#%% 
