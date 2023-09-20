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

from scipy.interpolate import interp1d

run_calculations='n'
#%%
def calving(t,Uc0):
    # seasonality in melt rate
    
    Uc = (0.1*np.sin(2*np.pi*t)+1)*Uc0
   

    return(Uc)

#%%
if run_calculations=='y':
    starting_files = sorted(glob.glob('*.pickle'))
    order = [2,1,0]
    starting_files = [starting_files[x] for x in order]
    
    t = 0
    
    directories = ['small','medium','large']
    for k in np.arange(1,2):#len(starting_files)):
        
        file = open(starting_files[k],'rb')
        data = pickle.load(file)
        file.close()
    
        data.transient = 1
        
        Uc = data.Uc
        Uc0 = Uc
        time = 0
    
        H0 = data.H0
        L = data.L
        F = glaciome1D.force(data)
        
        j = 1
        
        
        while t <= 5:
            data.dt = 0.01#*data.dx*data.L/data.U[0]
            t += data.dt
            data.Uc = calving(t,Uc0)
            data.prognostic()
            data.save(directories[k] + '/seasonality_' + "{:04d}".format(j) + '.pickle')
            
            #time = np.append(time, t)
            Uc = np.append(Uc, data.Uc)
            H0 = np.append(H0, data.H0)
            L = np.append(L, data.L)
            F = np.append(F, glaciome1D.force(data))
            
            print('Time: ' + "{:.04f}".format(t) + ' yr')
            print('dt: ' + "{:.05f}".format(data.dt) + ' yr')
            print('Uc: ' + "{:.04f}".format(data.Uc) + ' m/yr')
            print('H0: ' + "{:.02f}".format(data.H0) + ' m')
            print('CFL: ' + "{:.04f}".format(data.U[0]*data.dt/(data.dx*data.L)))
            print(' ')
            j += 1
            
        np.savez(directories[k] + '/seasonality.npz',time=time,Uc=Uc,H0=H0,L=L,F=F)

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


#dt = 0.01
#T = 5 # years
#n = int(T/dt) # number of time steps
#t = np.linspace(0,T,n+1)

lag = np.zeros(len(files))

for j in np.arange(0,len(files)):
    data = np.load(files[j])

    
    time = data['time']

    if j==1:
        ax1.plot(time,data['Uc'],'--',color=cmap(color_id[j]))
        ax1_twin.plot(time,data['F']*1e-7,'-',color=cmap(color_id[j]))
        
    # perform cross-correlation
    # need to introduce interpolation here...
    
    T = 5#time[-1] # simulation duration
    dt = 0.01 # time step for interpolation
    t = np.linspace(0,T,int(T/dt)+1,endpoint=True)
    Uc_interpolator = interp1d(time, data['Uc'], axis=-1)
    F_interpolator = interp1d(time, data['F'], axis=-1)
    
    Uc = Uc_interpolator(t)
    F = F_interpolator(t)

    
    x = Uc-np.mean(Uc)
    x = x/np.max(x)
    
    
    y = F-np.mean(F)
    y = y/np.max(y)
    
    correlation = correlate(x,y,mode='full')
    lags = correlation_lags(x.size,y.size,mode='full')
    lag[j] = lags[np.argmax(correlation)]*dt
    
    
    ax2.plot(data['Uc'],data['F']*1e-7, color=cmap(color_id[j]),label='{:.2f}'.format(-lag[j]) + ' a')
    
ax2.legend(loc='lower right',title='Lag time')    
    

#plt.savefig('fig-response_time.pdf',format='pdf',dpi=300)