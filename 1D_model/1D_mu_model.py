import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags

import scipy.integrate as integrate
from scipy.interpolate import interp1d

from config import *

#import muI
import granular_fluidity

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:27:08 2022

@author: jason
"""


#%%


dt = secsDay*1

B = 0/secsYear # mass balance rate in m/s

x0 = 0
L = 3e3+x0 # melange length
dx = 100 # grid spacing
x = np.arange(x0,L+dx,dx) # longitudinal grid
W = 5000*np.ones(x.shape) # melange width; treated as constant for now
muW = muS*np.ones(x.shape)


#H = (-200/L*x + 200) + d
#H = -50*(x/L-1) + d # initial melange thickness
H = np.ones(x.shape)*(d) # melange thickness

Ut = 10000/secsYear # glacier terminus velocity
U = Ut*(1+x/L) # initial guess for the velocity


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

n = 500 # number of time steps
color_id = np.linspace(0,1,n) 


for k in np.arange(0,n):
    print(k*dt/secsDay)     
    U, mu, muW = granular_fluidity.velocity(x,Ut,U,H,W,dx)
             
    
    dHdt = (B-np.gradient(H*W*U,x)/W)
    #dHdt[-1] = 0
    # update thickness
    H += dHdt*dt
    #H[:-1] += (B-np.diff(H*W*U)/W[:-1]/dx)*dt
    
            
    # # update grid
    # x0 = x[0]+Ut*dt
    # xL = x[-1]+U[-1]*dt

    # xg = np.linspace(x0,xL,len(x))
    # dx = x[1]-x[0]
    
    # # this interpolation needs some thought!
    # H_interp1d = interp1d(x, H, kind='linear', fill_value='extrapolate') #, right=d) # linear extrapolation???
    # H = H_interp1d(xg)
    
    # U_interp1d = interp1d(x, U, kind='linear', fill_value='extrapolate')
    # U = U_interp1d(xg)
    
    # W = np.interp(xg, x, W)
    # x = xg
    
    x = x+U*dt
    xg = np.linspace(x[0],x[-1],len(x))
    
    H_interp1d = interp1d(x,H,kind='linear', fill_value='extrapolate')
    U_interp1d = interp1d(x, U, kind='linear', fill_value='extrapolate')
    W_interp1d = interp1d(x, W, kind='linear', fill_value='extrapolate')
    
    H = H_interp1d(xg)
    U = U_interp1d(xg)
    W = W_interp1d(xg)
    
    x = xg
    

    if k % 10 == 0:
        ax1.plot(x,U*secsYear,color=plt.cm.viridis(color_id[k]))
        ax2.plot(x,H,color=plt.cm.viridis(color_id[k]))
        ax3.plot(x,mu,color=plt.cm.viridis(color_id[k]))
        ax4.plot(x,muW,color=plt.cm.viridis(color_id[k]))
plt.tight_layout()

