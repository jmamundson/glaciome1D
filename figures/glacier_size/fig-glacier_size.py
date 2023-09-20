#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 12:39:00 2023

@author: jason
"""

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
import matplotlib.patheffects as PathEffects
import sys

sys.path.append('/home/jason/projects/glaciome/glaciome1D')
from glaciome1D import constants, glaciome, pressure, transverse, deformational_thickness, force

import os

matplotlib.rc('lines',linewidth=1) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

import cmasher as cmr

cmap = cmr.get_sub_cmap('viridis', 0, 0.95)

import glob
import pickle

constant = constants()

#%% 
run_simulations = 'n'

if run_simulations == 'y':
    
    size = ['small', 'medium', 'large']
    terminus_velocity = np.array([0.6e4,0.65e4,0.7e4])
    calving_rate = terminus_velocity
    terminus_width = np.array([4800,5200,5600])
    terminus_thickness = np.array([600,650,700])
    
    
    
    for j in np.arange(0,len(size)):
    
        n_pts = 11 # number of grid points
        L = 1e4 # ice melange length
        Ut = terminus_velocity[j] # glacier terminus velocity [m/a]; treated as a constant
        Uc = calving_rate[j] # glacier calving rate [m/a]; treated as a constant
        Ht = terminus_thickness[j] # terminus thickness
        dt = 0.01# 1/(n_pts-1)/10 # time step [a]; needs to be quite small for this to work
    
        # specifying fjord geometry
        X_fjord = np.linspace(-200e3,200e3,101)
        
        W_fjord = terminus_width[j] + 0/10000*X_fjord
        
        data = glaciome(n_pts, dt, L, Ut, Uc, Ht, X_fjord, W_fjord)
        data.B = -0.6*constant.daysYear
        data.steadystate()
        data.refine_grid(21)
        data.steadystate()
        data.save('steady-state_' + size[j] + '_.pickle')


    
    


#%%   
def set_up_figure():
    '''
    Sets up the basic figure for plotting. 

    Returns
    -------
    axes handles ax1, ax2, ax3, ax4, ax5, and ax_cbar for the 5 axes and colorbar
    '''
    
    color_id = np.linspace(0,1,3)


    cm = 1/2.54
    fig_width = 18*cm
    fig_height = 12*cm
    plt.figure(figsize=(fig_width,fig_height))
    
    
    ax_width = 3.8*cm/fig_width
    ax_height = 3.8*cm/fig_height
    left = 2*cm/fig_width
    bot = 1.5*cm/fig_height
    ygap = 2*cm/fig_height
    xgap= 2*cm/fig_width
    
    
    xmax = 15
    vmax = 300
    
    text_pos_scale = 6.5/3.8
    
    ax1 = plt.axes([left, bot+ax_height+ygap, ax_width, ax_height])
    ax1.set_xlabel('Longitudinal coordinate [m]')
    ax1.set_ylabel('Speed [m/d]')
    ax1.set_ylim([0,vmax])
    ax1.set_xlim([0,xmax])
    ax1.set_yticks(np.linspace(0,300,4,endpoint=True))
    ax1.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'a',transform=ax1.transAxes,va='top',ha='left')
    
    
    ax2 = plt.axes([left+ax_width+xgap, bot+ax_height+ygap, ax_width, ax_height])
    ax2.set_xlabel('Longitudinal coordinate [km]')
    ax2.set_ylabel('Elevation [m]')
    ax2.set_ylim([-400, 100])
    ax2.set_xlim([0,xmax])
    txt = ax2.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'b',transform=ax2.transAxes,va='top',ha='left')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    
    
    ax3 = plt.axes([left, bot, ax_width, ax_height])
    ax3.set_xlabel('Longitudinal coordinate [km]')
    ax3.set_ylabel('$g^\prime$ [a$^{-1}]$')
    ax3.set_ylim([0, 40])
    ax3.set_xlim([0,xmax])
    txt = ax3.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'c',transform=ax3.transAxes,va='top',ha='left')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    #ax3.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'c',transform=ax3.transAxes,va='top',ha='left')
    
    
    ax4 = plt.axes([left+ax_width+xgap, bot, ax_width, ax_height])
    ax4.set_xlabel('Longitudinal coordinate [km]')
    ax4.set_ylabel('$\mu_w$')
    ax4.set_ylim([0, 2])
    ax4.set_xlim([0,xmax])
    ax4.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'d',transform=ax4.transAxes,va='top',ha='left')
    
    ax5 = plt.axes([left+2*(ax_width+xgap), bot, ax_width, 2*ax_height+ygap])
    ax5.set_xlabel('Transverse coordinate [km]')
    ax5.set_ylabel(r'Speed [m/d]')
    ax5.set_xlim([-2.8,2.8])
    ax5.set_ylim([0,vmax+50])
    txt = ax5.text(0.05*text_pos_scale,1-0.05*6.5/(2*3.8+2),'e',transform=ax5.transAxes,va='top',ha='left')
    

    axes = (ax1, ax2, ax3, ax4, ax5)
    
    return(axes, color_id)





#%%
def plot_figure(data, axes, color_id, linestyle):
    '''
    Take the current state of the model object and plot the basic figure.
    '''
    
    # extract variables from model object
    X = data.X
    X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
    U = data.U
    H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))
    W = data.W
    gg = np.concatenate(([1.5*data.gg[0]-0.5*data.gg[1]],data.gg,[1.5*data.gg[-1]-0.5*data.gg[-2]]))
    muW = np.concatenate(([3*data.muW[0]-3*data.muW[1]+data.muW[2]],data.muW,[3*data.muW[-1]-3*data.muW[-2]+data.muW[-3]]))
    
    W = data.W
    
    X = X-X[0]
    X_ = X_-X_[0]


    ax1, ax2, ax3, ax4, ax5 = axes
    ax1.plot(X*1e-3,U/constant.daysYear,color=plt.cm.viridis(color_id[0]),linestyle=linestyle)
    ax2.plot(np.append(X_,X_[::-1])*1e-3,np.append(-constant.rho/constant.rho_w*H,(1-constant.rho/constant.rho_w)*H[::-1]),color=plt.cm.viridis(color_id[0]),linestyle=linestyle)
    ax3.plot(X_*1e-3,gg,color=plt.cm.viridis(color_id[0]),linestyle=linestyle)
    ax4.plot(X*1e-3,muW,color=plt.cm.viridis(color_id[0]),linestyle=linestyle)
    
    
    # compute transverse velocity profile
    # need to clean this up later
    W = data.W[0]
    
    muW = data.muW[[0,10,-1]]
    U = data.U[[1,11,-2]]
    
    H= np.concatenate(([(data.H[0]+data.H[1])/2], [(data.H[10]+data.H[11])/2], [(data.H[-1]+data.H[-2])/2]))  
    
    #H = np.concatenate(([H[1]], np.mean([H[10],H[11]]), [H[-2]]))
    #H = np.concatenate(([data.H0],[np.mean([data.H[10],data.H[11]])],[data.HL]))
    
    #muW = data.muW[0]
    #H = H[0]
    #U = (U[0]+U[1])/2
    Hd = deformational_thickness(H,data)
    
    for j in np.arange(0,3):#len(Hd)):
        if data.subgrain_deformation=='n':
            #y, u_transverse, _ = transverse(W[ind],muW[ind],deformational_thickness(H[ind]))
            y, u_transverse, _ = transverse(W,muW[j],Hd[j],data)
    
        else:
            y, u_transverse, _ = transverse(W,muW[j],Hd[j],data)

        u_mean = np.trapz(u_transverse,y,y[1])/y[-1]/constant.daysYear      
        
        u_slip = U[j]-np.mean(u_transverse)
        u_transverse += u_slip
        
        ax5.plot(np.append(y-y[-1],y)*1e-3,np.append(u_transverse,u_transverse[-1::-1])/constant.daysYear,color=cmap(color_id[j]),linestyle=linestyle)

        ax5.legend(('$\chi=0.05$','$\chi=0.5$','$\chi=0.95$'))

#%%
files = sorted(glob.glob('./*.pickle'))
#file = files[1]
F = np.zeros(len(files))
Q = np.zeros(len(files))
axes, color_id = set_up_figure()
linestyle = ['dotted','--','-']
for j in np.arange(0,len(files)):
    with open(files[j], 'rb') as file:
        data = pickle.load(file)
        file.close()
    F[j] = force(data)
    Q[j] = data.Ht*data.Ut*data.W[0]
    plot_figure(data,axes,color_id,linestyle[j])
    if j==0:
        axes[1].plot(np.array([data.L,20000])*1e-3,np.array([0,0]),'k')
        
ax1, ax2, ax3, ax4, ax5 = axes
#glacier_x = np.array([-1000,0,0,-1000])
#glacier_y = np.array([-data.Ht*constant.rho/constant.rho_w,-data.Ht*constant.rho/constant.rho_w,data.Ht*(1-constant.rho/constant.rho_w),data.Ht*(1-constant.rho/constant.rho_w)+5])
#ax2.plot(glacier_x,glacier_y,'k')
#ax2.fill(np.array([-2000,13000,13000,-2000]),np.array([glacier_y[0],glacier_y[0],-600,-600]),'oldlace',edgecolor='k')

    
plt.savefig('fig_glacier_size.pdf',format='pdf',dpi=300)
