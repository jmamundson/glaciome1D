#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:04:35 2023

@author: jason
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import matplotlib.patheffects as PathEffects
import sys

sys.path.append('/hdd/glaciome/models/glaciome1D')
from glaciome1D import glaciome, constants

import os

from scipy.interpolate import interp1d

matplotlib.rc('lines',linewidth=1) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

import cmasher as cmr

cmap = cmr.get_sub_cmap('viridis', 0, 0.95)

import glob
import pickle

import shutil
constant = constants()

#%%

run_simulations = 'y'
terminus_width = np.linspace(3000,7000,5,endpoint=True)
dWdx = np.linspace(-0.04,0.04,5)

if run_simulations == 'y':
    
    shutil.copyfile('../fig1-steady_state_profile/steady-state_Bdot_-0.60.pickle','./steady-state_Wt_4000.0.pickle')
    
    n_pts = 51 # number of grid points
    L = 1e4 # ice melange length
    Ut = 0.6e4 # glacier terminus velocity [m/a]; treated as a constant
    Uc = 0.6e4 # glacier calving rate [m/a]; treated as a constant
    Ht = 600 # terminus thickness
    n = 101 # number of time steps
    dt = 0.01# 1/(n_pts-1)/10 # time step [a]; needs to be quite small for this to work
    B = -0.6*constant.daysYear
    
    for j in np.arange(0,len(terminus_width)): #np.array([4]):#np.array([0,2,3,4]):
        
        file = open('./steady-state_Wt_4000.0.pickle','rb')
        data = pickle.load(file)
        file.close()
        
        if j==4:
            file = open('./steady-state_Wt_6000.0.pickle','rb')
            data = pickle.load(file)
            file.close()
        
        # specifying fjord geometry
        data.Wt = terminus_width[j]
        data.W_fjord = data.Wt + 0/10000*data.X_fjord
            
        print('W_t: ' + "{0}".format(data.Wt))
        
        data.steadystate(method='lm')

        # data.steadystate(method='lm')
        data.save('steady-state_Wt_' + "{0}".format(data.Wt) + '.pickle')

    
    for j in np.arange(0,len(dWdx)):
    
        file = open('./steady-state_Wt_4000.0.pickle','rb')
        data = pickle.load(file)
        file.close()    
    
        # specifying fjord geometry
        data.Wt = data.W0
        data.W_fjord = data.Wt + dWdx[j]*data.X_fjord
        
        print('dW/dx: ' + "{0}".format(dWdx[j]))
        
        data.steadystate(method='lm')
        
        # data.steadystate(method='lm')
        data.save('steady-state_dWdx_' + "{0}".format(dWdx[j]) + '.pickle')
        
        

#%%


files_W = sorted(glob.glob('*Wt*.pickle'))


files_dWdx = sorted(glob.glob('*dWdx*.pickle'))
order = [1, 0, 2, 3, 4]
files_dWdx = [files_dWdx[x] for x in order]

cm = 1/2.54
fig_width = 18*cm
fig_height = 19.5*cm
plt.figure(figsize=(fig_width,fig_height))

left = 2*cm/fig_width
bottom = 1.5*cm/fig_height
width = 6.5*cm/fig_width
height = 6.5*cm/fig_height
cbar_height = 0.2*cm/fig_height
xgap = 2*cm/fig_width

ax1 = plt.axes([left,2*bottom+(height+2*bottom),width,height])
ax2 = plt.axes([left+width+xgap, 2*bottom+(height+2*bottom), width, height])
ax3 = plt.axes([left,2*bottom, width,height])
ax4 = plt.axes([left+width+xgap, 2*bottom, width, height])

ax1_cbar = plt.axes([left,bottom+(height+2*bottom),2*width+xgap,cbar_height])
ax3_cbar = plt.axes([left,bottom,2*width+xgap,cbar_height])

linestyles=['-','--']

# first files_W

n = len(files_W)
color_id = np.linspace(0,1,len(files_dWdx),endpoint=True)




#%%
H0 = np.zeros(len(files_W))
F = np.zeros(len(files_W))
W = np.zeros(len(files_W))
F_quasistatic =  np.zeros(len(files_W))

# k=0 # counter for cycling through colors...

for j in np.arange(0,len(files_W)):
   
    with open(files_W[j], 'rb') as file:
        data = pickle.load(file)
        file.close()
        
    
    H0[j] = data.H0
    F[j] = data.force()
    W[j] = data.W0
    F_quasistatic[j] = data.H0*data.pressure(data.H0)

    X = np.concatenate(([0],data.X_,[data.L]))
    X = np.concatenate((X,X[-1::-1]))
    H = np.concatenate(([data.H0],data.H,[data.HL]))
    H = np.concatenate((H*(1-constant.rho/constant.rho_w),-H[-1::-1]*constant.rho/constant.rho_w))
    
    ax1.plot(X,H,color=plt.cm.viridis(color_id[j]),linestyle=linestyles[0])
    
    
    if j==0:
        ax1.plot(np.array([data.L,20000]),np.array([0,0]),'k')

P0 = data.pressure(H0)


ax2.semilogy(W,F*1e-7,'k',linestyle=linestyles[1],zorder=1)
ax2.semilogy(W,F_quasistatic*1e-7,'k',linestyle='dotted',zorder=1)

for j in np.arange(0,len(W)):
    ax2.semilogy(W[j],F[j]*1e-7,'o',markersize=3,color=plt.cm.viridis(color_id[j]))
    ax2.semilogy(W[j],F_quasistatic[j]*1e-7,'o',fillstyle='none', markersize=3,color=plt.cm.viridis(color_id[j]))


# glacier_x = np.array([-1000,0,0,-1000])
# glacier_y = np.array([-data.Ht*constant.rho/constant.rho_w,-data.Ht*constant.rho/constant.rho_w,data.Ht*(1-constant.rho/constant.rho_w),data.Ht*(1-constant.rho/constant.rho_w)+5])
# #ax1.plot(glacier_x,glacier_y,'k')
# ax1.fill(np.array([-2000,20000,20000,-2000]),np.array([glacier_y[0],glacier_y[0],-600,-600]),'oldlace',edgecolor='k')


ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Elevation [m]')
ax1.set_xlim([0,20000])
ax1.set_ylim([-400,100])
txt = ax1.text(0.05,0.95,'a',transform=ax1.transAxes,va='top',ha='left')
txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])


ax2.set_xlabel('$W$ [m]')
ax2.set_ylabel(r'$F/W$ [$\times 10^{7}$ N m$^{-1}$]')
ax2.text(0.05,0.95,'b',transform=ax2.transAxes,va='top',ha='left')
ax2.set_xlim([2000,8000])
#ax2.set_xticks(np.linspace(0.4,1.1,8,endpoint=True))
#ax2.set_xticklabels(['0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1'])
ax2.set_ylim([0.01,10])
ax2.set_yticks((np.logspace(-2,1,4,base=10)))#,np.linspace(1,10,11,endpoint=True))))
#ax2.set_yticks(np.array([0.1,1,10]))
ax2.set_yticklabels(['0.01','0.1','1','10'])#'2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])#np.linspace(0.2,1.0,9,endpoint=True))

cbar_ticks = np.linspace(3000, 7000, 5, endpoint=True)
cmap = matplotlib.cm.viridis
bounds = cbar_ticks
norm = matplotlib.colors.Normalize(vmin=3000, vmax=7000)
cb = matplotlib.colorbar.ColorbarBase(ax1_cbar, cmap=cmap, norm=norm,
                                orientation='horizontal')#,extend='min')
cb.set_label('$W$ [m]')

#%%
H0 = np.zeros(len(files_dWdx))
F = np.zeros(len(files_dWdx))
dWdx = np.zeros(len(files_dWdx))
F_quasistatic = np.zeros(len(files_dWdx))

for j in np.arange(0,len(files_dWdx)):

    with open(files_dWdx[j], 'rb') as file:
        data = pickle.load(file)
        file.close()
        
    H0[j] = data.H0
    F[j] = data.force()
    dWdx[j] = (data.WL-data.W0)/data.L
    F_quasistatic[j] = data.H0*data.pressure(data.H0)

    X = np.concatenate(([0],data.X_,[data.L]))
    X = np.concatenate((X,X[-1::-1]))
    H = np.concatenate(([data.H0],data.H,[data.HL]))
    H = np.concatenate((H*(1-constant.rho/constant.rho_w),-H[-1::-1]*constant.rho/constant.rho_w))
    
    ax3.plot(X,H,color=plt.cm.viridis(color_id[j]),linestyle=linestyles[0])
    ax4.semilogy(dWdx[j],F[j]*1e-7,'o',markersize=3,color=plt.cm.viridis(color_id[j]))
    
    
    if j==0:
        ax3.plot(np.array([data.L,20000]),np.array([0,0]),'k')


ax4.semilogy(dWdx,F*1e-7,'k',linestyle=linestyles[1],zorder=1)
ax4.semilogy(dWdx,F_quasistatic*1e-7,'k',linestyle='dotted',zorder=1)

for j in np.arange(0,len(dWdx)):
    ax4.semilogy(dWdx[j],F[j]*1e-7,'o',markersize=3,color=plt.cm.viridis(color_id[j]))
    ax4.semilogy(dWdx[j],F_quasistatic[j]*1e-7,'o',fillstyle='none', markersize=3,color=plt.cm.viridis(color_id[j]))



glacier_x = np.array([-1000,0,0,-1000])
glacier_y = np.array([-data.Ht*constant.rho/constant.rho_w,-data.Ht*constant.rho/constant.rho_w,data.Ht*(1-constant.rho/constant.rho_w),data.Ht*(1-constant.rho/constant.rho_w)+5])
#ax1.plot(glacier_x,glacier_y,'k')
ax3.fill(np.array([-2000,20000,20000,-2000]),np.array([glacier_y[0],glacier_y[0],-600,-600]),'oldlace',edgecolor='k')


ax3.set_xlabel('Longitudinal coordinate [m]')
ax3.set_ylabel('Elevation [m]')
ax3.set_xlim([0,20000])
ax3.set_ylim([-400,100])
txt = ax3.text(0.05,0.95,'c',transform=ax3.transAxes,va='top',ha='left')
txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])


ax4.set_xlabel('$dW/dx$')
ax4.set_ylabel(r'$F/W$ [$\times 10^{7}$ N m$^{-1}$]')
ax4.text(0.05,0.95,'d',transform=ax4.transAxes,va='top',ha='left')
ax4.set_xlim([-0.05,0.05])
#ax2.set_xticks(np.linspace(0.4,1.1,8,endpoint=True))
#ax2.set_xticklabels(['0.4','0.5','0.6','0.7','0.8','0.9','1.0','1.1'])
# ax4.set_ylim([0.1,10])
ax4.set_yticks((np.logspace(-1,1,3,base=10)))#,np.linspace(1,10,11,endpoint=True))))
#ax2.set_yticks(np.array([0.1,1,10]))
ax4.set_yticklabels(['0.1','1','10'])#'2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0'])#np.linspace(0.2,1.0,9,endpoint=True))

cbar_ticks = np.linspace(-0.04, 0.04, 5, endpoint=True)
cmap = matplotlib.cm.viridis
bounds = cbar_ticks
norm = matplotlib.colors.Normalize(vmin=-0.04, vmax=0.04)
cb = matplotlib.colorbar.ColorbarBase(ax3_cbar, cmap=cmap, norm=norm,
                                orientation='horizontal')#,extend='min')
cb.set_label('$dW/dx$')




plt.savefig('fig-varyW.pdf',format='pdf',dpi=300)