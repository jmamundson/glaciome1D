#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:38:07 2023

@author: jason
"""
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

cmap = cmr.get_sub_cmap('viridis', 0, 0.8)

from glaciome1D import constants
constant = constants()
#%%


files_B = sorted(glob.glob('seasonality_B/*/*npz'))
file_order = [1,2,0]
files_B = [files_B[x] for x in file_order]

n = len(files_B)
color_id = np.linspace(0,1,n,endpoint=True)

cm = 1/2.54
fig_width = 18*cm
fig_height = 16.5*cm
plt.figure(figsize=(fig_width,fig_height))

left = 2*cm/fig_width
bottom = 1.5*cm/fig_height
width = 6.5*cm/fig_width
height = 6.5*cm/fig_height
cbar_height = 0.2*cm/fig_height
xgap = 2.1*cm/fig_width
ygap = 1.5*cm/fig_height

ax1 = plt.axes([left, bottom+height+ygap, width,height])
ax2 = plt.axes([left+width+xgap, bottom+height+ygap, width, height])
ax3 = plt.axes([left, bottom, width, height])
ax4 = plt.axes([left+width+xgap, bottom, width, height])

ax1.set_xlim([1.75,3.75])
ax1.set_xticks(np.linspace(1.75,3.75,5,endpoint=True))
ax1.set_xticklabels(np.linspace(0,2,5,endpoint=True))
ax1.set_ylim([0.1,1.1])
ax1.set_yticks(np.linspace(0.1,1.1,6,endpoint=True))
#ax1.set_yticks(np.linspace(-1.4,-0.2,7,endpoint=True))
ax1.set_xlabel('Time [a]')
ax1.set_ylabel('Melt rate [m d$^{-1}$]')
ax1_twin = ax1.twinx()
ax1_twin.set_ylim([0,6])
ax1_twin.set_ylabel('\n $\Delta F$ [$10^7$ N/m]')
ax1.text(0.05,1-0.05,'a',transform=ax1.transAxes,va='top',ha='left')

ax2.set_xlim([0.1,1.1])
ax2.set_xticks(np.linspace(0.1,1.1,6,endpoint=True))
ax2.set_xlabel(r'Melt rate [m d$^{-1}$]')
ax2.set_ylim([0,6])
ax2.text(0.05,1-0.05,'b',transform=ax2.transAxes,va='top',ha='left')

ax3.set_xlim([1.75,3.75])
ax3.set_xticks(np.linspace(1.75,3.75,5,endpoint=True))
ax3.set_xticklabels(np.linspace(0,2,5,endpoint=True))
ax3.set_ylim([4000,8000])
ax3.set_yticks(np.linspace(4000,8000,5,endpoint=True))
ax3.set_xlabel('Time [a]')
ax3.set_ylabel('Calving rate [m a$^{-1}$]')
ax3_twin = ax3.twinx()
ax3_twin.set_ylim([0,6])
ax3_twin.set_ylabel('\n $\Delta F$ [$10^7$ N/m]')
ax3.text(0.05,1-0.05,'c',transform=ax3.transAxes,va='top',ha='left')

ax4.set_xlim([4000,8000])
ax4.set_xlabel(r'Calving rate [m a$^{-1}$]')
ax4.set_ylim([0,6])
ax4.text(0.05,1-0.05,'d',transform=ax4.transAxes,va='top',ha='left')

### melting only
dt = 0.01
T = 5 # years
n = int(T/dt) # number of time steps
t = np.linspace(0,T,n+1)

lag = np.zeros(len(files_B))

for j in np.arange(0,len(files_B)):
    data = np.load(files_B[j])

    if j==2:
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
    

    
ax2.legend(loc='upper right',title='Lag time')    
    


### calving only
files_calving = sorted(glob.glob('./seasonality_calving/*/*npz'))
lag = np.zeros(len(files_calving))

color_id = np.linspace(0,1,len(files_calving))
for j in np.arange(0,len(files_calving)):
    data = np.load(files_calving[j])

    if j==2:
        ax3.plot(t,data['Uc'],'--',color=cmap(color_id[j]))
        ax3_twin.plot(t,data['F']*1e-7,'-',color=cmap(color_id[j]))
        
    # perform cross-correlation
    x = data['Uc']-np.mean(data['Uc'])
    x = x/np.max(x)
    
    y = data['F']-np.mean(data['F'])
    y = y/np.max(y)
    
    correlation = correlate(x,y,mode='full')
    lags = correlation_lags(x.size,y.size,mode='full')
    lag[j] = lags[np.argmax(correlation)]*dt
    
    
    ax4.plot(data['Uc'],data['F']*1e-7, color=cmap(color_id[j]),label='{:.2f}'.format(-lag[j]) + ' a')
    
ax4.legend(loc='upper right',title='Lag time') 

### calving coupled to melt
file_coupled = 'seasonality_B_calving/output/seasonality.npz'
data = np.load(file_coupled)

#ax1.plot(data['t'],-data['B']/constant.daysYear,'--k')
ax1_twin.plot(data['t'],data['F']*1e-7,'k')
ax2.plot(-data['B']/constant.daysYear,data['F']*1e-7,'k',)
ax3_twin.plot(data['t'],data['F']*1e-7,'k')
ax3.plot(data['t'],data['Uc'],'--k')


ax2.annotate("", xy=(0.3, 3), xytext=(0.5, 2.9), arrowprops=dict(arrowstyle="->"))
ax4.annotate("", xy=(6600, 1.8), xytext=(7400, 1.9), arrowprops=dict(arrowstyle="->"))

plt.savefig('fig-seasonality.pdf',format='pdf',dpi=300)

# correlate coupled model
# perform cross-correlation
x = data['B']-np.mean(data['B'])
x = x/np.max(x)

#y = data['H0']-np.mean(data['H0']) 
#y = y/np.max(y)

y = data['F']-np.mean(data['F'])
y = y/np.max(y)

correlation = correlate(x,y,mode='full')
lags = correlation_lags(x.size,y.size,mode='full')
print(lags[np.argmax(correlation)]*dt)
