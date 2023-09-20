#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 08:49:13 2023

@author: jason
"""


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

sys.path.append('/home/jason/projects/glaciome/glaciome1D')
import glaciome1D
import matplotlib.patheffects as PathEffects

from glaciome1D import constants
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

import cmasher as cmr

cmap = cmr.get_sub_cmap('viridis', 0, 0.95)

constant = constants()

run_calculations='y'
#%%
def Bdot(t,Bdot0):
    # seasonality in melt rate
    
    Bdot = -0.2*np.sin(2*np.pi*t)+Bdot0
    Bdot = Bdot*constant.daysYear
    
    return(Bdot)

#%%
if run_calculations=='n':
    starting_files = sorted(glob.glob('*.pickle'))
    
    directories = ['low','medium','high','veryhigh'] # melt rates are low, medium, or high
    for k in np.arange(3,len(starting_files)):
        file = open(starting_files[k],'rb')
        data = pickle.load(file)
        file.close()
        
        Bdot0 = data.B/constant.daysYear
        
        data.transient = 1
        
        data.dt = 0.01
        T = 5 # years
        n = int(T/data.dt) # number of time steps
        
        t = np.linspace(0,T,n+1,endpoint=True)
        
        #data.save('seasonality-0pt6/seasonality_000.pickle')
        
        
        B = [Bdot(t,Bdot0) for t in t]
        H0 = np.zeros(n+1)
        L = np.zeros(n+1)
        F = np.zeros(n+1)
        
        H0[0] = data.H0
        L[0] = data.L
        F[0] = glaciome1D.force(data)
        
        for j in np.arange(0,n):
            print('Time: ' + str(j*data.dt) + ' yr')
            data.B = Bdot(t[j],Bdot0)
            data.prognostic()
            data.save(directories[k] + '/seasonality_' + "{:04d}".format(j) + '.pickle')
            H0[j+1] = data.H0
            L[j+1] = data.L
            F[j+1] = glaciome1D.force(data)
            print('H0: ' + "{:02f}".format(data.H0))
        
        np.savez(directories[k] + '/seasonality.npz',B=B,H0=H0,L=L,F=F)

#%%
files = sorted(glob.glob('*/*npz'))
order = [0,2,1]
files = [files[x] for x in order]

n = len(files)
color_id = np.linspace(0,1,n,endpoint=True)

cm = 1/2.54
fig_width = 18*cm
fig_height = 8.5*cm
plt.figure(figsize=(fig_width,fig_height))

left = 2*cm/fig_width
bottom = 1.5*cm/fig_height
width = 6.5*cm/fig_width
height = 6.5*cm/fig_height
cbar_height = 0.2*cm/fig_height
xgap = 1.9*cm/fig_width

ax1 = plt.axes([left,bottom,width,height])
ax2 = plt.axes([left+width+xgap*1.1, bottom, width, height])


ax1.set_xlim([2,4])
ax1.set_xticks(np.linspace(2,4,5,endpoint=True))
ax1.set_xticklabels(np.linspace(0,2,5,endpoint=True))
ax1.set_ylim([0.1,1.1])
ax1.set_yticks(np.linspace(0.1,1.1,6,endpoint=True))
#ax1.set_yticks(np.linspace(-1.4,-0.2,7,endpoint=True))
ax1.set_xlabel('Time [a]')
ax1.set_ylabel('Melt rate [m d$^{-1}$]')
ax1_twin = ax1.twinx()
ax1_twin.set_ylim([0,8])
ax1_twin.set_ylabel('\n $\Delta F$ [$10^7$ N/m]')
ax1.text(0.05,1-0.05,'a',transform=ax1.transAxes,va='top',ha='left')



ax2.set_xlim([0.1,1.1])
ax2.set_xticks(np.linspace(0.1,1.1,6,endpoint=True))
ax2.set_xlabel(r'Melt rate [m d$^{-1}$]')
ax2.set_ylim([0,8])
ax2.text(0.05,1-0.05,'b',transform=ax2.transAxes,va='top',ha='left')


dt = 0.01
T = 5 # years
n = int(T/dt) # number of time steps
t = np.linspace(0,T,n+1)

lag = np.zeros(len(files))

for j in np.arange(0,len(files)):
    data = np.load(files[j])

    if j==1:
        ax1.plot(t,-data['B']/constant.daysYear,'--',color=cmap(color_id[j]))
        ax1_twin.plot(t,data['F']*1e-7,'-',color=cmap(color_id[j]))
        
    # perform cross-correlation
    x = data['B']-np.mean(data['B'])
    x = x/np.max(x)
    
    #y = data['H0']-np.mean(data['H0']) 
    #y = y/np.max(y)
    
    y = data['F']-np.mean(data['F'])
    y = y/np.max(y)
    
    correlation = correlate(x,y,mode='full')
    lags = correlation_lags(x.size,y.size,mode='full')
    lag[j] = lags[np.argmax(correlation)]*dt
    
    
    ax2.plot(-data['B']/constant.daysYear,data['F']*1e-7, color=cmap(color_id[j]),label='{:.2f}'.format(-lag[j]) + ' a')
    

    #plt.plot(data['B']/constant.daysYear,data['L'])
    
    #ax1.plot(t,-data['B']/constant.daysYear)
    #ax2.plot(t,data['H0'])
ax2.legend(loc='upper right',title='Lag time')    
    




plt.savefig('fig-response_time.pdf',format='pdf',dpi=300)