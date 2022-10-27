import numpy as np
from matplotlib import pyplot as plt

from scipy.optimize import root

from general_utilities import second_invariant

from config import *

import importlib

# here you should specify the rheology that should be imported
rheology = 'granular_fluidity'
#rheology = 'muI'

model = importlib.import_module(rheology)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jason
"""


#%

dt = secsDay*5 # time step [s]
n = 21 # number of time steps

B = 0/secsYear # mass balance rate [m s^-1]

x0 = 0 # left boundary of the melange [m]
L = 10000 # initial ice melange length [m]
dx = 0.1    # grid spacing [m]
x = np.arange(0,1+dx,dx) # longitudinal grid
X = x*L # coordinates of unstretched grid

W = 5000*np.ones(len(x)-1) # fjord width [m]; treated as constant for now
H = np.ones(len(x)-1)*d # initial ice melange thickness [m]


Ut = 10000/secsYear # width-averaged glacier terminus velocity [m/s]
U = Ut*(1-x) # initial guess for the averaged velocity [m/s]; the model
# can go haywire if this isn't chosen carefully...


# determine velocity profile that is consistent with initial thickness; unlike 
# subsequent steps this does not involve an implicit time step
U = root(model.spinup, U, (x,X,Ut,H,W,dx), method='hybr', options={'xtol':1e-12})
#U = U.x

#%%
plt.figure(figsize=(12,8))
ax1 = plt.subplot(221)
ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Speed [m/yr]')
ax1.set_ylim([-1000,11000])
ax1.set_xlim([0,10000])

ax2 = plt.subplot(222)
ax2.set_xlabel('Longitudinal coordinate [m]')
ax2.set_ylabel('Thickness [m]')
ax2.set_ylim([0, 100])
ax2.set_xlim([0,10000])

ax3 = plt.subplot(223)
ax3.set_xlabel('Longitudinal coordinate [m]')
ax3.set_ylabel('$\mu$')
ax3.set_ylim([0.1, 0.7])
ax3.set_xlim([0,10000])

ax4 = plt.subplot(224)
ax4.set_xlabel('Longitudinal coordinate [m]')
ax4.set_ylabel('$\mu_w$')
ax4.set_ylim([0.1, 0.7])
ax4.set_xlim([0,10000])

color_id = np.linspace(0,1,n) 


# plot initial time step
ax1.plot(X,U*secsYear,color=plt.cm.viridis(color_id[0]))
ax2.plot((X[:-1]+X[1:])/2,H,color=plt.cm.viridis(color_id[0]))

ee_chi = second_invariant(U,dx)
#mu, muW = model.get_mu(x,U,H,W,dx,ee_chi,L)

#ax3.plot((X[:-1]+X[1:])/2,mu,color=plt.cm.viridis(color_id[0]))
#ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[0]))

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
    X = np.linspace(xt,xL,len(x))
    
    
    # ee_chi = second_invariant(U,dx)
    # mu, muW = model.get_mu(x,U,H,W,dx,ee_chi,X[-1]-X[0])
    
    
    ax1.plot(X,U*secsYear,color=plt.cm.viridis(color_id[k]))
    ax2.plot((X[:-1]+X[1:])/2,H,color=plt.cm.viridis(color_id[k]))
    # ax3.plot((X[:-1]+X[1:])/2,mu,color=plt.cm.viridis(color_id[k]))
    # ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[k]))

plt.tight_layout()
plt.savefig('test.png',format='png',dpi=300)



    

#     if k % 1 == 0:
#         ax1.plot(x,U*secsYear,color=plt.cm.viridis(color_id[k]))
#         ax2.plot(x,H,color=plt.cm.viridis(color_id[k]))
#         ax3.plot(x,mu,color=plt.cm.viridis(color_id[k]))
#         ax4.plot(x,muW,color=plt.cm.viridis(color_id[k]))
# plt.tight_layout()

