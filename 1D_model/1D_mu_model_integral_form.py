import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags

import scipy.integrate as integrate
from scipy.interpolate import interp1d
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:27:08 2022

@author: jason
"""

# global vs local variables!




#%%
secsDay = 86400
secsYear = secsDay*365.25
dt = secsDay*1

rho = 917 # density of ice
rho_w = 1028 # density of water

g = 9.81 # gravity

B = 0/secsYear # mass balance rate in m/s

d = 25 # characteristic iceberg size

x0 = 0
L = 0.5e4+x0 # melange length
dx = 100 # grid spacing
x = np.arange(x0,L+dx,dx) # longitudinal grid
W = 1000*np.ones(x.shape) # melange width; treated as constant for now

#H = (-200/L*x + 200) + d
#H = -200*(x/L-1) + d # initial melange thickness
H = np.ones(x.shape)*(d) # melange thickness
Ut = 10000/secsYear # glacier terminus velocity


# max and min effective coefficients of friction, sort of
mu0 = 0.6
muS = 0.2
I0 = 10**-7


ax1 = plt.subplot(311)
ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Speed [m/yr]')
ax2 = plt.subplot(312)
ax2.set_xlabel('Longitudinal coordinate [m]')
ax2.set_ylabel('Thickness [m]')
ax3 = plt.subplot(313)
ax3.set_xlabel('Longitudinal coordinate [m]')
ax3.set_ylabel('$\mu$')


n = 1000 # number of time steps
color_id = np.linspace(0,1,n) 

#%%
for k in np.arange(0,n):
    integrand = (2*H-d)*np.gradient(H,dx)+2*0.1*mu0/W*H**2
    integral = integrate.cumtrapz(-integrand[::-1],x=x[::-1])[::-1]
    integral = np.append(integral,0)
    P = 1/2*rho*g*(1-rho/rho_w)*H
    
    mu = -(1/H**2)*integral
    sgn_exx = np.sign(mu) # 1 => exx>0; -1 => exx<0
    
    mu = np.abs(mu) # mu must be positive
    mu[ mu<muS ] = muS # no deformation beneath muS (yield stress not reached)
    mu[ mu>=mu0 ] = mu0-0.01
    
    I = I0*(muS-mu)/(mu-mu0)
    ee = I*np.sqrt(P/rho)/d
    dUdx = ee*sgn_exx
    
    U = Ut + np.append(0, integrate.cumtrapz(dUdx,x=x))

    dHdt = (B-np.gradient(H*W*U,x)/W)
    H += dHdt*dt


    xL = x[-1]+(U[-1]-Ut)*dt # advance the end of the melange
    x0 = x[0] # advance terminus
    
    xg = np.linspace(x0,xL,len(x)) # new grid
    
    #x += Ut*dt # advance entire melange at terminus flow speed;
               # this way there is no interpolation at the terminus boundary
    
    xg = np.linspace(x0,xL,len(x))
    
    H_interp1d = interp1d(x, H, kind='linear', fill_value='extrapolate') #, right=d) # linear extrapolation???
    H = H_interp1d(xg)
    
    U_interp1d = interp1d(x, U, kind='linear', fill_value='extrapolate')
    U = U_interp1d(xg)
    
    W = np.interp(xg, x, W)
    x = xg
    
    if k % 100 == 0:
        ax1.plot(x, U*secsYear,color=plt.cm.viridis(color_id[k]))
        ax2.plot(x,H,color=plt.cm.viridis(color_id[k]))
        ax3.plot(x,mu,color=plt.cm.viridis(color_id[k]))

plt.tight_layout()