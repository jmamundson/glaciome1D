#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 19:15:01 2023

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
from glaciome1D import glaciome, pressure

import config
from scipy.integrate import quad


matplotlib.rc('lines',linewidth=1) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

n_pts = 11 # number of grid points
L = 1e4 # ice melange length
Ut = 0.5e4 # glacier terminus velocity [m/a]; treated as a constant
Uc = 0.4e4 # glacier calving rate [m/a]; treated as a constant
Ht = 500 # terminus thickness
n = 101 # number of time steps
dt = 0.01 # time step [a]; needs to be quite small for this to work

# specifying fjord geometry
X_fjord = np.linspace(-200e3,200e3,101)
W_fjord = 4000*np.ones(len(X_fjord))

data = glaciome(n_pts, dt, L, Ut, Uc, Ht, X_fjord, W_fjord)

B = np.linspace(-0.5,-1,6,endpoint=True)*config.daysYear

tauX = np.array([0,4000]) # drag force per unit width

H0 = np.zeros(len(B))


#%%
n = len(B)
color_id = np.linspace(0,1,n)

cm = 1/2.54
fig_width = 18*cm
fig_height = 10*cm
plt.figure(figsize=(fig_width,fig_height))

left = 2*cm/fig_width
bottom = 1.5*cm/fig_height
width = 6.5*cm/fig_width
height = 6.5*cm/fig_height
cbar_height = 0.2*cm/fig_height
xgap = 2*cm/fig_width

ax1 = plt.axes([left,2*bottom,width,height])
ax2 = plt.axes([left+width+xgap, 2*bottom, width, height])

ax_cbar = plt.axes([left,bottom,width,cbar_height])

linestyles=['-','--']

#%%
for k in np.arange(0,len(tauX)):
    print('k: ' + str(k))
    data.tauX = tauX[k]
    for j in np.arange(0,len(B)):
        print('j: ' + str(j))
        data.B = B[j]
        data.steadystate()
        H0[j] = data.H0
    
    
        X = np.concatenate(([0],data.X_,[data.L]))
        X = np.concatenate((X,X[-1::-1]))
        H = np.concatenate(([data.H0],data.H,[data.HL]))
        H = np.concatenate((H*(1-config.rho/config.rho_w),-H[-1::-1]*config.rho/config.rho_w))
        
        ax1.plot(X,H,color=plt.cm.viridis(color_id[j]),linestyle=linestyles[k])
    
    P0 = pressure(H0)
    
    ax2.loglog(-B/config.daysYear,P0*H0*1e-7,'k',linestyle=linestyles[k])

ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Elevation [m]')
ax1.set_xlim([0,12000])
ax1.set_ylim([-120-140*0.05,20+140*0.05])
ax1.text(0.05,0.95,'a',transform=ax1.transAxes,va='top',ha='left')

ax2.set_xlabel('Melt rate [m d$^{-1}$]')
ax2.set_ylabel(r'$F/W$ [$\times 10^{-7}$ N m$^{-1}$]')
ax2.text(0.05,0.95,'b',transform=ax2.transAxes,va='top',ha='left')
ax2.set_xlim([0.4,1.1])
ax2.set_xticks(np.linspace(0.4,1.1,8,endpoint=True))
ax2.set_xticklabels(['0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1'])
ax2.set_ylim([0.2,1])
ax2.set_yticks(np.linspace(0.2,1,9,endpoint=True))
ax2.set_yticklabels(['0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])#np.linspace(0.2,1.0,9,endpoint=True))

cbar_ticks = np.linspace(100, 500, 5, endpoint=True)
cmap = matplotlib.cm.viridis
bounds = cbar_ticks
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1)
cb = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm,
                                orientation='horizontal')#,extend='min')
cb.set_label('Melt rate [m d$^{-1}$]')



plt.savefig('fig_steady-state_vary_ocean.pdf',format='pdf',dpi=300)
#plt.close()
#%%