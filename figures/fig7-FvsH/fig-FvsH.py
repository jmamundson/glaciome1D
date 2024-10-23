#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:04:35 2023

@author: jason
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import sys

sys.path.append('/hdd/glaciome/models/glaciome1D')
import glaciome1D
import matplotlib.patheffects as PathEffects


from scipy.integrate import quad
from scipy.signal import correlate     
from scipy.signal import correlation_lags

matplotlib.rc('lines',linewidth=1) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

import glob
import pickle
import os
import cmasher as cmr

cmap = cmr.get_sub_cmap('viridis', 0, 0.8)

from glaciome1D import constants
constant = constants()


#%%
def set_up_figure():
    '''
    Sets up the basic figure for plotting. 

    Returns
    -------
    '''

    cm = 1/2.54
    fig_width = 9*cm
    fig_height = 5.8*cm
    plt.figure(figsize=(fig_width,fig_height))
    
    
    ax_width = 3.8*1.6*cm/fig_width
    ax_height = 3.8*cm/fig_height
    left = 2*cm/fig_width
    bot = 1.5*cm/fig_height
    
    ax = plt.axes([left, bot, ax_width, ax_height])
    ax.set_xlabel(r'$H_0$ [m]')
    ax.set_ylabel(r'$F/W$ [$\times 10^{7}$ N m$^{-1}$]')
    
    
    

    return(ax)

#%%
ax = set_up_figure()

files_B = sorted(glob.glob('../fig6-seasonality/*varyB/*npz')) 
files_calving = sorted(glob.glob('../fig6-seasonality/*varyUc/*npz'))
file_coupled = '../fig6-seasonality/medium_melt_rate_varyBandUc/seasonality.npz'

color_id = np.linspace(0,1,3)

H = np.linspace(0,500)
ax.loglog(H, 1e-7*0.5*constant.rho*constant.g*(1-constant.rho/constant.rho_w)*H**2, 'k', zorder=1000, label='quasi-static')

n = 3 # use every nth data point

for j in files_B:
    data = np.load(j)
    ax.loglog(data['H0'][::n], data['F'][::n]*1e-7, '.', alpha=0.5, markersize=2, color=cmap(color_id[1]))

for j in files_calving:
    data = np.load(j)
    ax.loglog(data['H0'][::n], data['F'][::n]*1e-7, '.', alpha=0.5, markersize=2, color=cmap(color_id[1]))

data = np.load(file_coupled)
ax.loglog(data['H0'][::n], data['F'][::n]*1e-7, '.', alpha=0.5, markersize=2, color=cmap(color_id[1]))


files_steadystate = sorted(glob.glob('../fig4-vary_B/*pickle'))
H0 = np.zeros(len(files_steadystate))
F = np.zeros(len(files_steadystate))
for j in np.arange(0,len(files_steadystate)):
    file = open(files_steadystate[j],'rb')
    data = pickle.load(file)
    file.close()
    H0[j] = data.H0
    F[j] = data.force()*1e-7

ax.loglog(H0,F,'k--',label='steady-state')

ax.set_xlim([70, 170])
ax.set_xticks(ticks=np.linspace(70,170,6),labels=np.linspace(70,170,6,dtype='int'))
ax.set_ylim([0.2,1.4])
ax.set_yticks(ticks=np.linspace(0.2,1.4,7),labels=np.around(np.linspace(0.2,1.4,7),decimals=1))
ax.minorticks_off()

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)

plt.savefig('fig-FvsH.pdf',format='pdf',dpi=300)