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
from glaciome1D import glaciome, constants
import matplotlib.patheffects as PathEffects



matplotlib.rc('lines',linewidth=1) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

import glob
import pickle

constant = constants()

import cmasher as cmr

cmap = cmr.get_sub_cmap('viridis', 0, 0.95)

#%%
run_simulations = 'y'


if run_simulations == 'y':
    
    B_days = np.linspace(-1.0, -0.5, 6, endpoint=True)
    
    B = B_days*constant.daysYear
    
    n_pts = 51 # number of grid points
    L = 1e4 # ice melange length
    Ut = 0.6e4 # glacier terminus velocity [m/a]; treated as a constant
    Uc = 0.6e4 # glacier calving rate [m/a]; treated as a constant
    Ht = 600 # terminus thickness
    n = 101 # number of time steps
    dt = 0.01# 1/(n_pts-1)/10 # time step [a]; needs to be quite small for this to work

    # specifying fjord geometry
    X_fjord = np.linspace(-200e3,200e3,101)
    Wt = 4000
    W_fjord = Wt + 0/10000*X_fjord
    
    data = glaciome(n_pts, dt, L, Ut, Uc, Ht, B[0], X_fjord, W_fjord)
    

    for j in np.arange(2,len(B)):
            
        data.B = B[j]
        data.steadystate(method='lm')
    
        data.save('steady-state_Bdot_' + "{:.2f}".format(B_days[j]) + '.pickle')




#%%
files = sorted(glob.glob('*.pickle'))
#files = files[10:]
#files = files[::5]


n = len(files)
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

ax_cbar = plt.axes([left,bottom,2*width+xgap,cbar_height])

linestyles=['-','--']

B = np.zeros(len(files))
H0 = np.zeros(len(files))
F = np.zeros(len(files))
F_quasistatic = np.zeros(len(files))

#%%
for j in np.arange(0,len(files)):
    with open(files[j], 'rb') as file:
        data = pickle.load(file)
        file.close()
    
    
    B[j] = data.B
    H0[j] = data.H0
    F[j] = data.force()
    F_quasistatic[j] = data.H0*data.pressure(data.H0)

    X = np.concatenate(([0],data.X_,[data.L]))
    X = np.concatenate((X,X[-1::-1]))
    H = np.concatenate(([data.H0],data.H,[data.HL]))
    H = np.concatenate((H*(1-constant.rho/constant.rho_w),-H[-1::-1]*constant.rho/constant.rho_w))
    
    ax1.plot(X,H,color=plt.cm.viridis(color_id[j]),linestyle=linestyles[0])
    if j==0:
        ax1.plot(np.array([data.L,20000]),np.array([0,0]),'k')

    
ax2.loglog(-B/constant.daysYear,F*1e-7,'k',linestyle=linestyles[1])
ax2.loglog(-B/constant.daysYear,F_quasistatic*1e-7,'k',linestyle='dotted')

for j in np.arange(0,len(B)):
    ax2.loglog(-B[j]/constant.daysYear,F[j]*1e-7,'o',color=plt.cm.viridis(color_id[j]),markersize=3)
    ax2.loglog(-B[j]/constant.daysYear,F_quasistatic[j]*1e-7,marker='o', fillstyle='none',color=plt.cm.viridis(color_id[j]),markersize=3)


#ax2.semilogy(-B[4]/constant.daysYear,F[4]*1e-7,'ko',fillstyle='none',markersize=6)

glacier_x = np.array([-1000,0,0,-1000])
glacier_y = np.array([-data.Ht*constant.rho/constant.rho_w,-data.Ht*constant.rho/constant.rho_w,data.Ht*(1-constant.rho/constant.rho_w),data.Ht*(1-constant.rho/constant.rho_w)+5])
#ax1.plot(glacier_x,glacier_y,'k')
ax1.fill(np.array([-2000,20000,20000,-2000]),np.array([glacier_y[0],glacier_y[0],-600,-600]),'oldlace',edgecolor='k')

ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Elevation [m]')
ax1.set_xlim([0,18000])
ax1.set_ylim([-300,100])
txt = ax1.text(0.05,0.95,'a',transform=ax1.transAxes,va='top',ha='left')
txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

ax2.set_xlabel('Melt rate [m d$^{-1}$]')
ax2.set_ylabel(r'$F/W$ [$\times 10^{7}$ N m$^{-1}$]')
ax2.text(0.05,0.95,'b',transform=ax2.transAxes,va='top',ha='left')
ax2.set_xlim([0.4,1.1])
ax2.set_xticks(np.linspace(0.4,1.1,8,endpoint=True))
ax2.set_xticklabels(['0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1'])
ax2.set_ylim([0.1,10])
#ax2.set_yticks(np.concatenate((np.linspace(0.1,0.9,10,endpoint=True),np.linspace(1,10,11,endpoint=True))))
ax2.set_yticks(np.array([0.1,1,10]))
ax2.set_yticklabels(['0.1','1','10'])#'2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])#np.linspace(0.2,1.0,9,endpoint=True))



cbar_ticks = np.linspace(100, 500, 5, endpoint=True)
cmap = matplotlib.cm.viridis
bounds = cbar_ticks
norm = matplotlib.colors.Normalize(vmin=0.5, vmax=1.0)
cb = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm,
                                orientation='horizontal')#,extend='min')
cb.set_label('Melt rate [m d$^{-1}$]')



plt.savefig('fig-steady-state_melt_rates.pdf',format='pdf',dpi=300)
#plt.close()
#%%