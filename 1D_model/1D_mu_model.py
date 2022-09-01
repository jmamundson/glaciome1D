import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags

import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import minimize

from config import *

import importlib

# here you should specify the rheology that should be imported
#rheology = 'granular_fluidity'
rheology = 'muI'

model = importlib.import_module(rheology)


#import granular_fluidity

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:27:08 2022

@author: jason
"""


#%%

dt = secsDay*1 # time step [s]

B = 0/secsYear # mass balance rate [m s^-1]

x0 = 0 # left boundary of the melange [m]
L = 5e3+x0 # initial ice melange length [m]
dx = 100 # grid spacing [m]
x = np.arange(x0,L+dx,dx) # longitudinal grid
W = 5000*np.ones(x.shape) # fjord width [m]; treated as constant for now

H = np.ones(x.shape)*d # initial ice melange thickness [m]
#H = d*(1-x/L)+d # initial ice melange thickness [m]

Ut = 10000/secsYear # width-averaged glacier terminus velocity [m s^-1]
U = Ut*(1-x/L) # initial guess for the width-averaged velocity [m s^-1]

mu = np.zeros(len(x))
muW = np.zeros(len(x))

#U = np.zeros(x.shape)
#U[0] = Ut

#%%
plt.figure(figsize=(12,8))
ax1 = plt.subplot(221)
ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Speed [m/yr]')
ax1.set_ylim([0,10000])

ax2 = plt.subplot(222)
ax2.set_xlabel('Longitudinal coordinate [m]')
ax2.set_ylabel('Thickness [m]')
ax2.set_ylim([0, 100])

ax3 = plt.subplot(223)
ax3.set_xlabel('Longitudinal coordinate [m]')
ax3.set_ylabel('$\mu$')
ax3.set_ylim([0.1, 0.7])

ax4 = plt.subplot(224)
ax4.set_xlabel('Longitudinal coordinate [m]')
ax4.set_ylabel('$\mu_w$')
ax4.set_ylim([0.1, 0.7])

n = 51 # number of time steps
color_id = np.linspace(0,1,n) 


for k in np.arange(0,n):
    print('Time: ' + "{:.0f}".format(k*dt/secsDay) + ' days')     
    #U, mu, muW = model.velocity(U,x,Ut,H,W,dx)
             
    U = fsolve(model.velocity, U, (x,Ut,H,W,dx), xtol=0.1/secsYear)
    
    mu, muW = model.get_mu(x,U,H,W,dx)
    
    
    #np.savez('./results/time_' + "{:03d}".format(int(k*dt/secsDay)) + '.npz'  , x=x, H=H, U=U, W=W)#, mu=mu, muW=muW)
    
    
    # UPDATE THE ICE MELANGE THICKNESS USING MASS CONTINUITY
    # 1) use central differences to calculate rate of thickness change; 
    # second order accurate at the boundaries (using
    # ghost points?)
    # 2) update the thickness using an explicit time step
    dHdt = (B-np.gradient(H*W*U, x, edge_order=1)/W) # rate of thickness change
    H += dHdt*dt # new thickness
        
    # UPDATE THE GRID
    # 1) move each grid point forward using the calculated velocity
    # 2) create a new grid
    # 3) interpolate variables to the new, evenly spaced grid
    #L = L + (U[-1]-Ut)*dt
    
    x = x+U*dt
    xg = np.linspace(x[0],x[-1],len(x))
    
    #xg = np.linspace(x[0],L,len(x))
    
    H_interp1d = interp1d(x,H,kind='linear', fill_value='extrapolate')
    U_interp1d = interp1d(x, U, kind='linear', fill_value='extrapolate')
    W_interp1d = interp1d(x, W, kind='linear', fill_value='extrapolate')
    
    H = H_interp1d(xg)
    U = U_interp1d(xg)
    W = W_interp1d(xg)
    
    x = xg
    

    if k % 1 == 0:
        ax1.plot(x,U*secsYear,color=plt.cm.viridis(color_id[k]))
        ax2.plot(x,H,color=plt.cm.viridis(color_id[k]))
        ax3.plot(x,mu,color=plt.cm.viridis(color_id[k]))
        ax4.plot(x,muW,color=plt.cm.viridis(color_id[k]))
plt.tight_layout()

