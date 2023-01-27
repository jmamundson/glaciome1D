import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from scipy.optimize import root

from general_utilities import second_invariant, width

from config import *

import importlib

# here you should specify the rheology that should be imported
rheology = 'granular_fluidity'
model = importlib.import_module(rheology)

# changing A and b doesn't do anything. Why!!!!

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jason
"""


#%

dt = secsDay*10 # time step [s]
n = 21 # number of time steps

B = 0/secsYear # mass balance rate [m s^-1]

x0 = 0 # left boundary of the melange [m]
L = 10000 # initial ice melange length [m]
dx = 0.2   # grid spacing [m]
x = np.arange(0,1+dx,dx) # longitudinal grid
X = x*L # coordinates of unstretched grid

 
W = width((X[:-1]+X[1:])/2)
#W = 4000*np.ones(len(x)-1) # fjord width [m]; treated as constant for now
# W needs to move with the grid...

H = np.ones(len(x)-1)*d # initial ice melange thickness [m]


Ut = 10000/secsYear # width-averaged glacier terminus velocity [m/s]
U = Ut*(1-x) # initial guess for the averaged velocity [m/s]; the model



# determine velocity profile that is consistent with initial thickness; unlike 
# subsequent steps this does not involve an implicit time step
U = root(model.spinup, U, (x,X,Ut,H,W,dx,dt), method='lm', options={'xtol':1e-6})
U = U.x

#%%

fig_width = 12
fig_height = 8
plt.figure(figsize=(12,8))

ax_width = 4.75/fig_width
ax_height = 2.75/fig_height
left = 1/fig_width
bot = 0.5/fig_height
ygap = 0.75/fig_height
xgap= 1/fig_width

ax1 = plt.axes([left, bot+ax_height+2.25*ygap, ax_width, ax_height])
ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Speed [m/yr]')
ax1.set_ylim([-1000,11000])
ax1.set_xlim([0,13000])

ax2 = plt.axes([left+ax_width+xgap, bot+ax_height+2.25*ygap, ax_width, ax_height])
ax2.set_xlabel('Longitudinal coordinate [m]')
ax2.set_ylabel('Thickness [m]')
ax2.set_ylim([0, 100])
ax2.set_xlim([0,13000])

ax3 = plt.axes([left, bot+1.25*ygap, ax_width, ax_height])
ax3.set_xlabel('Longitudinal coordinate [m]')
ax3.set_ylabel('$\mu$')
ax3.set_ylim([0, 0.9])
ax3.set_xlim([0,13000])

ax4 = plt.axes([left+ax_width+xgap, bot+1.25*ygap, ax_width, ax_height])
ax4.set_xlabel('Longitudinal coordinate [m]')
ax4.set_ylabel('$\mu_w$')
ax4.set_ylim([0.1, 0.9])
ax4.set_xlim([0,13000])

ax_cbar = plt.axes([left, bot, 2*ax_width+xgap, ax_height/15])

cbar_ticks = np.linspace(0, (n-1)*dt/secsDay, 11, endpoint=True)
cmap = matplotlib.cm.viridis
bounds = cbar_ticks
norm = matplotlib.colors.Normalize(vmin=0, vmax=(n-1)*dt/secsDay)
cb = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm,
                                orientation='horizontal')#,extend='min')
cb.set_label("Time [d]")

#%%


color_id = np.linspace(0,1,n) 


# plot initial time step
ax1.plot(X,U*secsYear,color=plt.cm.viridis(color_id[0]))
ax2.plot((X[:-1]+X[1:])/2,H,color=plt.cm.viridis(color_id[0]))

mu, muW = model.get_mu(x,U,H,W,X[-1]-X[0],dx)

ax3.plot((X[:-1]+X[1:])/2,mu,color=plt.cm.viridis(color_id[0]))
ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[0]))

# concatenate U and H since the implicit time step requires that they are
# iteratively solved simultaneously
UH = np.append(U,H)


for k in np.arange(1,n):
    print('Time: ' + "{:.0f}".format(k*dt/secsDay) + ' days')     

    UH = root(model.convergence, UH, (x,X,Ut,H,W,dx,dt,U,H,B), method='hybr', options={'xtol':1e-6})
    UH = UH.x

    # Note: I think all that needs to be saved are UH, W, and the initial fjord length
    
    # the following is for plotting purposes
    U = UH[:len(x)]
    H = UH[len(x):]

    xt = X[0] + U[0]*dt
    xL = X[-1] + U[-1]*dt
    X = np.linspace(xt,xL,len(x))-xt
    
    mu, muW = model.get_mu(x,U,H,W,X[-1]-X[0],dx)
    
    W = width((X[:-1]+X[1:])/2)
    
    
    ax1.plot(X,U*secsYear,color=plt.cm.viridis(color_id[k]))
    ax2.plot((X[:-1]+X[1:])/2,H,color=plt.cm.viridis(color_id[k]))
    ax3.plot((X[:-1]+X[1:])/2,mu,color=plt.cm.viridis(color_id[k]))
    ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[k]))


