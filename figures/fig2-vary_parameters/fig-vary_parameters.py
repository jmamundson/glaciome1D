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
from glaciome1D import glaciome, basic_figure, plot_basic_figure, constants
import matplotlib.patheffects as PathEffects



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


#%% 
run_simulations = 'n'

if run_simulations == 'y':
    
    n_pts = 21 # number of grid points
    L = 1e4 # ice melange length
    Ut = 0.6e4 # glacier terminus velocity [m/a]; treated as a constant
    Uc = 0.6e4 # glacier calving rate [m/a]; treated as a constant
    Ht = 600 # terminus thickness
    n = 101 # number of time steps
    dt = 0.01
    
    # specifying fjord geometry
    X_fjord = np.linspace(-200e3,200e3,101)
    Wt = 4000
    W_fjord = Wt + 0/10000*X_fjord
    
    file_extensions = ['','_varymuS','_varyd','_varyb','_varyA']

    for j in np.arange(2,3):    
        # file = open('steady-state_Bdot_-0.80_31pts.pickle','rb')
        # data = pickle.load(file)
        # file.close()
        
        data = glaciome(n_pts, dt, L, Ut, Uc, Ht, X_fjord, W_fjord)
        data.B = -0.6*data.constants.daysYear
        
        if j==1:
            muS = 0.2
            print('muS: ' + "{:.03f}".format(muS))
            data.param.muS = muS
            
        if j==2:
            d = 10
            print('d: ' + "{:.03f}".format(d))
            data.param.d = d
            data.H -= 15
            data.H0 = 1.5*data.H[-1] - 0.5*data.H[-2]
            data.HL = 1.5*data.H[0] - 0.5*data.H[1]
            data.dt = 0.001
            
        if j==3:
            b = 5e4
            print('b: ' + "{:.03f}".format(b))
            data.param.b = b
            
        if j==4:
            A = 5
            print('A: ' + "{:.03f}".format(A))
            data.param.A = A
        
        data.diagnostic() # initialize with diagnostic solve
        data.steadystate(method='hybr')
        
        data.save('steady-state_Bdot_-0.60_' + file_extensions[j] + '.pickle')    
    
    
        
#%%    
def set_up_figure():
    '''
    Sets up the basic figure for plotting. 

    Returns
    -------
    axes handles ax1, ax2, ax3, ax4, ax5, and ax_cbar for the 5 axes and colorbar
    '''
    
    color_id = np.linspace(0,1,5)


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
    vmax = 300
    
    text_pos_scale = 6.5/3.8
    
    ax1 = plt.axes([left, bot+ax_height+ygap, ax_width, ax_height])
    ax1.set_xlabel('Longitudinal coordinate [m]')
    ax1.set_ylabel('Speed [m/d]')
    ax1.set_ylim([0,vmax])
    ax1.set_xlim([0,xmax])
    ax1.set_yticks(np.linspace(0,vmax,5,endpoint=True))
    ax1.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'a',transform=ax1.transAxes,va='top',ha='left')
    
    
    ax2 = plt.axes([left+ax_width+xgap, bot+ax_height+ygap, ax_width, ax_height])
    ax2.set_xlabel('Longitudinal coordinate [km]')
    ax2.set_ylabel('Elevation [m]')
    ax2.set_ylim([-300, 100])
    ax2.set_xlim([0,xmax])
    txt = ax2.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'b',transform=ax2.transAxes,va='top',ha='left',zorder=1000)
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
    
    # ax3 = plt.axes([left, bot+ax_height+ygap, ax_width, ax_height])
    # ax3.set_xlabel('Longitudinal coordinate [m]')
    # ax3.set_ylabel('Speed [m/d]')
    # ax3.set_ylim([0,vmax])
    # ax3.set_xlim([0,xmax*2])
    # ax3.set_yticks(np.linspace(0,vmax,5,endpoint=True))
    # ax3.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'c',transform=ax1.transAxes,va='top',ha='left')
    
    # ax4 = plt.axes([left+ax_width+xgap, bot+ax_height+ygap, ax_width, ax_height])
    # ax4.set_xlabel('Longitudinal coordinate [km]')
    # ax4.set_ylabel('Elevation [m]')
    # ax4.set_ylim([-600, 100])
    # ax4.set_xlim([0,xmax*2])
    # txt = ax2.text(0.05*text_pos_scale,1-0.05*text_pos_scale,'d',transform=ax2.transAxes,va='top',ha='left',zorder=1000)
    # txt.set_path_effects([PathEffects.withStroke(linewidth=2, foreground='w')])
    
    ax5 = plt.axes([left+2*(ax_width+xgap), bot, ax_width, 2*ax_height+ygap])
    ax5.set_xlabel('Transverse coordinate [km]')
    ax5.set_ylabel(r'Speed at $\chi=0.5$ [m/d]')
    ax5.set_xlim([-2.4,2.4])
    ax5.set_ylim([0,vmax])
    txt = ax5.text(0.05*text_pos_scale,1-0.05*6.5/(2*3.8+2),'e',transform=ax5.transAxes,va='top',ha='left')
    

    axes = (ax1, ax2, ax3, ax4, ax5)
    
    return(axes, color_id)


#%%
def plot_figure(data, axes, color, linestyle, j, zorder):
    '''
    Take the current state of the model object and plot the basic figure.
    '''
    
    # extract variables from model object
    X = data.X
    X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
    U = data.U + (data.Ut-data.Uc)
    H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))
    W = data.W
    gg = np.concatenate(([1.5*data.gg[0]-0.5*data.gg[1]],data.gg,[1.5*data.gg[-1]-0.5*data.gg[-2]]))
    muW = data.muW
    
    
    X = X-X[0]
    X_ = X_-X_[0]


    ax1, ax2, ax3, ax4, ax5 = axes
    ax1.plot(X*1e-3,(U+(data.Ut-data.Uc))/constant.daysYear,color=plt.cm.viridis(color),linestyle=linestyle,zorder=zorder)
    ax2.plot(np.append(X_,X_[::-1])*1e-3,np.append(-constant.rho/constant.rho_w*H,(1-constant.rho/constant.rho_w)*H[::-1]),color=plt.cm.viridis(color),linestyle=linestyle,zorder=zorder)
    ax3.plot(X_*1e-3,gg,color=plt.cm.viridis(color),linestyle=linestyle,zorder=zorder)
    ax4.plot(X*1e-3,muW,color=plt.cm.viridis(color),linestyle=linestyle,zorder=zorder)
    
    
    y, u_transverse, u_mean = data.transverse(0.5, dimensionless=False) 
    
    U_ind = np.interp(0.5,data.x,data.U)
    u_slip = U_ind-np.mean(u_transverse)
    u_transverse += u_slip + (data.Ut-data.Uc)
    
    ax5.plot(np.append(y-y[-1],y)*1e-3,np.append(u_transverse,u_transverse[-1::-1])/constant.daysYear,color=cmap(color),linestyle=linestyle,zorder=zorder)

    

    #ax5.legend(('$\chi=0.05$','$\chi=0.5$','$\chi=0.95$'))

#%%
files = sorted(glob.glob('./*.pickle'))

#order = [5,8,7,6,9]
#order = [0,3,2,1,4]
#order = [4,7,6,5,8]


#files = [files[x] for x in order]


axes, color_id = set_up_figure()


linestyle = ['dotted','--','-']
for j in np.arange(0,len(files)):
    with open(files[j], 'rb') as file:
        data = pickle.load(file)
        file.close()
        
        if j==0:
            plot_figure(data,axes,color_id[j],linestyle[2],j,100)
            
        elif (j%2)==0:
            plot_figure(data,axes,color_id[j],linestyle[1],j,1) 
        else:
            plot_figure(data,axes,color_id[j],linestyle[0],j,1)
        
        if j==3:
            print(data.L)
            axes[1].plot(np.array([data.L/1000,20]),np.array([0,0]),color='k')

        


axes[4].legend(['default', r'$A=5$', r'$b=5\times 10^4$', r'$d=10$ m', r'$\mu_s=0.25$'],framealpha=1)

ax1, ax2, ax3, ax4, ax5 = axes
glacier_x = np.array([-1000,0,0,-1000])
glacier_y = np.array([-data.Ht*constant.rho/constant.rho_w,-data.Ht*constant.rho/constant.rho_w,data.Ht*(1-constant.rho/constant.rho_w),data.Ht*(1-constant.rho/constant.rho_w)+5])
ax2.plot(glacier_x,glacier_y,'k')
ax2.fill(np.array([-2000,17000,17000,-2000]),np.array([glacier_y[0],glacier_y[0],-600,-600]),'oldlace',edgecolor='k')


plt.savefig('fig-vary_parameters.pdf',format='pdf',dpi=300)
