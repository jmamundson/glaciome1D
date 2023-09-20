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

run_calculations='n'
#%%
def Bdot(t):
    # seasonality in melt rate
    
    Bdot = -0.2*np.sin(2*np.pi*t)-0.6
    Bdot = Bdot*constant.daysYear
    
    return(Bdot)

def calving(F,F0,Uc0):
    
    m = -Uc0/F0 #(1.1*Uc0-Uc0)/(1.1*F0-F0)
    b = 2*Uc0
    
    Uc = m*F + b
    
    return(Uc)

#%%
if run_calculations=='y':
    starting_files = sorted(glob.glob('*.pickle'))
    
    directories = ['output']
    for k in np.arange(0,len(starting_files)):
        file = open(starting_files[k],'rb')
        data = pickle.load(file)
        file.close()
    
        data.transient = 1
    
        data.dt = 0.01
        T = 10 # years
        n = int(T/data.dt) # number of time steps
    
        t = np.linspace(0,T,n+1,endpoint=True)
    
    
        B = [Bdot(t) for t in t]
        H0 = np.zeros(n+1)
        L = np.zeros(n+1)
        F = np.zeros(n+1)
        Uc = np.zeros(n+1)
        
        F0 = glaciome1D.force(data)
        Uc0 = data.Uc
    
        H0[0] = data.H0
        L[0] = data.L
        F[0] = glaciome1D.force(data)
        Uc[0] = data.Uc
    
        for j in np.arange(0,n):
            print('Time: ' + str(j*data.dt) + ' yr')
            
            data.prognostic()
            data.save(directories[k] + '/seasonality_' + "{:04d}".format(j+1) + '.pickle')
            
            data.B = B[j+1]
            H0[j+1] = data.H0
            L[j+1] = data.L
            F[j+1] = glaciome1D.force(data)
            Uc[j+1] = calving(F[j+1],F0,Uc0)
            data.Uc = Uc[j+1]
            
            print('H0: ' + "{:.02f}".format(H0[j+1]) + ' m')
            print('Uc: ' + "{:.02f}".format(Uc[j+1]) + ' m/a')
        
        np.savez(directories[k] + '/seasonality.npz',t=t,B=B,Uc=Uc,H0=H0,L=L,F=F)

#%%
files = sorted(glob.glob('*/*npz'))

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
ax1.set_ylim([4000,8000])
#ax1.set_yticks(np.linspace(-1.4,-0.2,7,endpoint=True))
ax1.set_xlabel('Time [a]')
ax1.set_ylabel('Calving rate [m a$^{-1}$]')
ax1_twin = ax1.twinx()
ax1_twin.set_ylim([0,5])
ax1_twin.set_ylabel('\n $\Delta F$ [$10^7$ N/m]')
ax1.text(0.05,1-0.05,'a',transform=ax1.transAxes,va='top',ha='left')



ax2.set_xlim([4000,8000])
ax2.set_xlabel(r'Calving rate [m a$^{-1}$]')
ax2.set_ylim([0,5])
ax2.text(0.05,1-0.05,'b',transform=ax2.transAxes,va='top',ha='left')


dt = 0.01
T = 10 # years
n = int(T/dt) # number of time steps
t = np.linspace(0,T,n+1)

lag = np.zeros(len(files))

for j in np.arange(0,len(files)):
    data = np.load(files[j])

    if j==0:
        ax1.plot(t,data['Uc'],'--',color=cmap(color_id[j]))
        ax1_twin.plot(t,data['F']*1e-7,'-',color=cmap(color_id[j]))
        
    # perform cross-correlation
    x = data['B']-np.mean(data['B'])
    x = x/np.max(x)
    
    
    
    y = data['F']-np.mean(data['F'])
    y = y/np.max(y)
    
    correlation = correlate(x,y,mode='full')
    lags = correlation_lags(x.size,y.size,mode='full')
    lag[j] = lags[np.argmax(correlation)]*dt
    
    
    ax2.plot(data['Uc'],data['F']*1e-7, color=cmap(color_id[j]),label='{:.2f}'.format(-lag[j]) + ' a')
    
ax2.legend(loc='lower right',title='Lag time')    
    

#plt.savefig('fig-response_time.pdf',format='pdf',dpi=300)

#%%
# plt.subplot(311)
# plt.plot(t,B)
# plt.xlim([2,4])

# plt.subplot(312)
# plt.plot(t,F)
# plt.xlim([2,4])

# plt.subplot(313)
# plt.plot(t,Uc)
# plt.xlim([2,4])