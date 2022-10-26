import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags

import scipy.integrate as integrate
from scipy.interpolate import interp1d
from scipy.optimize import fsolve
from scipy.optimize import minimize
from scipy.optimize import root
from scipy.optimize import newton_krylov

from general_utilities import second_invariant

from config import *

import importlib

# here you should specify the rheology that should be imported
#rheology = 'granular_fluidity3'
#rheology = 'muI'
rheology = 'muI_stretched'

model = importlib.import_module(rheology)


#import granular_fluidity

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:27:08 2022

@author: jason
"""


##%%

dt = secsDay*5 # time step [s]

B = 0/secsYear # mass balance rate [m s^-1]

x0 = 0 # left boundary of the melange [m]
L = 10000 # initial ice melange length [m]
dx = 0.1    # grid spacing [m]
x = np.arange(0,1+dx,dx) # longitudinal grid
X = x*L # coordinates of unstretched grid

W = 5000*np.ones(len(x)-1) # fjord width [m]; treated as constant for now
H = np.ones(len(x)-1)*d # initial ice melange thickness [m]
#H = d+20*(1-(x[:-1]+x[1:])/2)


Ut = 10000/secsYear # width-averaged glacier terminus velocity [m s^-1]
U = Ut*(1-x) # initial guess for the width-averaged velocity [m s^-1]
U[0] = Ut



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

n = 21 # number of time steps
color_id = np.linspace(0,1,n) 


#result = newton_krylov(lambda U:model.spinup(U,x,X,Ut,H,W,dx), U)
U = root(model.spinup, U, (x,X,Ut,H,W,dx), method='hybr', options={'xtol':1e-24})
U = U.x
ax1.plot(X,U*secsYear,color=plt.cm.viridis(color_id[0]))
ax2.plot((X[:-1]+X[1:])/2,H,color=plt.cm.viridis(color_id[0]))

ee_chi = np.sqrt(np.gradient(U,x)**2)+dee
mu, muW = model.get_mu(x,U,H,W,dx,ee_chi[:-1],L)

ax3.plot((X[:-1]+X[1:])/2,mu,color=plt.cm.viridis(color_id[0]))
ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[0]))

UH = np.append(U,H)

for k in np.arange(1,n):
    print('Time: ' + "{:.0f}".format(k*dt/secsDay) + ' days')     

    UH = root(model.velocity, UH, (x,X,Ut,H,W,dx,dt,U,H), method='hybr', options={'xtol':1e-6})#, options={'disp':True})#, 'fatol':1/secsYear})#, options={'ftol':1/secsYear})
    #U = root(model.velocity, U, (x,Ut,H,W,dx), method='lm', options={'xtol':1e-6})#, options={'disp':True})#, 'fatol':1/secsYear})#, options={'ftol':1/secsYear})
    
    UH = UH.x

    U = UH[:len(x)]
    H = UH[len(x):]

    xt = X[0] + U[0]*dt
    xL = X[-1] + U[-1]*dt
    X = np.linspace(xt,xL,len(x))
    
    
    ee_chi = second_invariant(U,X[1]-X[0])
    mu, muW = model.get_mu(x,U,H,W,dx,ee_chi,L)
    
    ax1.plot(X,U*secsYear,color=plt.cm.viridis(color_id[k]))
    ax2.plot((X[:-1]+X[1:])/2,H,color=plt.cm.viridis(color_id[k]))
    ax3.plot((X[:-1]+X[1:])/2,mu,color=plt.cm.viridis(color_id[k]))
    ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[k]))

plt.savefig('test.png',format='png',dpi=300)
    # define U_prev, H_prev
#     mu, muW = model.get_mu(x,U,H,W,dx)
    
    
#     np.savez('./results/time_' + "{:03d}".format(int(k*dt/secsDay)) + '.npz'  , x=x, H=H, U=U, W=W, mu=mu, muW=muW)#, mu=mu, muW=muW)
    
    
#     # UPDATE THE ICE MELANGE THICKNESS USING MASS CONTINUITY
#     # 1) use central differences to calculate rate of thickness change; 
#     # second order accurate at the boundaries (using
#     # ghost points?)
#     # 2) update the thickness using an explicit time step
#     dHdt = (B-np.gradient(H*W*U, x, edge_order=1)/W) # rate of thickness change
#     H += dHdt*dt # new thickness
        
#     # UPDATE THE GRID
#     # 1) move each grid point forward using the calculated velocity
#     # 2) create a new grid
#     # 3) interpolate variables to the new, evenly spaced grid
#     #L = L + (U[-1]-Ut)*dt
    
#     x = x+U*dt
#     xg = np.linspace(x[0],x[-1],len(x))
    
#     #xg = np.linspace(x[0],L,len(x))
    
#     H_interp1d = interp1d(x,H,kind='linear', fill_value='extrapolate')
#     U_interp1d = interp1d(x, U, kind='linear', fill_value='extrapolate')
#     W_interp1d = interp1d(x, W, kind='linear', fill_value='extrapolate')
    
#     H = H_interp1d(xg)
#     U = U_interp1d(xg)
#     W = W_interp1d(xg)
    
#     x = xg
    

#     if k % 1 == 0:
#         ax1.plot(x,U*secsYear,color=plt.cm.viridis(color_id[k]))
#         ax2.plot(x,H,color=plt.cm.viridis(color_id[k]))
#         ax3.plot(x,mu,color=plt.cm.viridis(color_id[k]))
#         ax4.plot(x,muW,color=plt.cm.viridis(color_id[k]))
# plt.tight_layout()

