#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 08:08:45 2023

@author: jason
"""

import numpy as np
from matplotlib import pyplot as plt
import pickle

import sys
sys.path.append('/home/jason/projects/glaciome/glaciome1D')
from glaciome1D import constants, glaciome

import matplotlib.patheffects as PathEffects

import glob

constant = constants()

import cmasher as cmr

cmap = cmr.get_sub_cmap('viridis', 0, 0.95)

#%%    
def set_up_figure():
    '''
    Sets up the basic figure for plotting. 

    Returns
    -------
    axes handles ax1, ax2, ax3, ax4, ax5, and ax_cbar for the 5 axes and colorbar
    '''



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
    
    
    xmax = 20
    vmax = 150
    
    text_pos_scale = 6.5/3.8
    
    ax1 = plt.axes([left, bot+ax_height+ygap, ax_width, ax_height])
    ax1.set_xlabel('Longitudinal coordinate [m]')
    ax1.set_ylabel('Speed [m/d]')
    ax1.set_ylim([0,vmax])
    ax1.set_xlim([0,xmax])
    ax1.set_yticks(np.linspace(0,vmax,4,endpoint=True))
    txt = ax1.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'a',transform=ax1.transAxes,va='top',ha='left')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    
    ax2 = plt.axes([left+ax_width+xgap, bot+ax_height+ygap, ax_width, ax_height])
    ax2.set_xlabel('Longitudinal coordinate [km]')
    ax2.set_ylabel('Elevation [m]')
    ax2.set_ylim([-300, 100])
    ax2.set_xlim([0,xmax])
    ax2.set_yticks(np.linspace(-300,100,5,endpoint=True))
    txt = ax2.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'b',transform=ax2.transAxes,va='top',ha='left')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    
    
    ax3 = plt.axes([left, bot, ax_width, ax_height])
    ax3.set_xlabel('Longitudinal coordinate [km]')
    ax3.set_ylabel('$g^\prime$ [a$^{-1}]$')
    ax3.set_ylim([0, 10])
    ax3.set_xlim([0,xmax])
    txt = ax3.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'c',transform=ax3.transAxes,va='top',ha='left')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])

    
    ax4 = plt.axes([left+ax_width+xgap, bot, ax_width, ax_height])
    ax4.set_xlabel('Longitudinal coordinate [km]')
    ax4.set_ylabel('$\mu_w$')
    ax4.set_ylim([0, 1])
    ax4.set_xlim([0,xmax])
    ax4.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'d',transform=ax4.transAxes,va='top',ha='left')
    
    ax5 = plt.axes([left+2*(ax_width+xgap), bot, ax_width, 2*ax_height+ygap])
    ax5.set_xlabel('Transverse coordinate [km]')
    ax5.set_ylabel(r'Speed [m/d]')
    ax5.set_xlim([-2.4,2.4])
    ax5.set_ylim([0,vmax])
    ax5.set_yticks(np.linspace(0,vmax,4,endpoint=True))
    txt = ax5.text(0.05*text_pos_scale,1-0.05*6.5/(2*3.8+2),'e',transform=ax5.transAxes,va='top',ha='left')
    txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])


    axes = (ax1, ax2, ax3, ax4, ax5)
    
    return(axes)

#%%
def plot_figure(data, axes, color, linestyle):
    '''
    Take the current state of the model object and plot the basic figure.
    '''
    
    # extract variables from model object
    X = data.X
    X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
    U = data.U
    H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))
    gg = np.concatenate(([1.5*data.gg[0]-0.5*data.gg[1]],data.gg,[1.5*data.gg[-1]-0.5*data.gg[-2]]))
    muW = data.muW# np.concatenate(([3*data.muW[0]-3*data.muW[1]+data.muW[2]],data.muW,[3*data.muW[-1]-3*data.muW[-2]+data.muW[-3]]))
    
    X = X-X[0]
    X_ = X_-X_[0]


    ax1, ax2, ax3, ax4, ax5 = axes
    ax1.plot(X*1e-3,(U+data.Ut-data.Uc)/constant.daysYear,color=color,linestyle=linestyle)
    ax2.plot(np.append(X_,X_[::-1])*1e-3,np.append(-constant.rho/constant.rho_w*H,(1-constant.rho/constant.rho_w)*H[::-1]),color=color,linestyle=linestyle)
    ax3.plot(X_*1e-3,gg,color=color,linestyle=linestyle)
    ax4.plot(X*1e-3,muW,color=color,linestyle=linestyle)
    
    
    
    
    y, u_transverse, u_mean = data.transverse(0.5,dimensionless=False)
    U_ind = np.interp(0.5,data.x,data.U)
    
    u_slip = U_ind-u_mean#np.mean(u_transverse)
    u_transverse += u_slip + (data.Ut - data.Uc)
    
    ax5.plot(np.append(y-y[-1],y)*1e-3,np.append(u_transverse,u_transverse[-1::-1])/constant.daysYear,color=color,linestyle=linestyle)

    

#%%

files = sorted(glob.glob('medium_melt_rate_varyB/*pickle'))

files = files[125:176:10]

color_id = np.linspace(0,1,len(files))

axes = set_up_figure()

for j in np.arange(0,len(files)):
    file = open(files[j], 'rb')
    data = pickle.load(file)
    file.close()
    print(data.B)
    color=cmap(color_id[j])
    
    plot_figure(data, axes, color, '-')