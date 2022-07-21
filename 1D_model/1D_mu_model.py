import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags

import scipy.integrate as integrate
from scipy.interpolate import interp1d

from config import *

import muI


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 11:27:08 2022

@author: jason
"""


    
#%%
def velocity(x,U,H,W,dx):
    
    # U is the initial guess for the velocity
    
    j = 1
    while j==1:

        nu, mu = muI.mu(x,U,H,dx)
        #nu, mu = granular_fluidity(x,U,H,dx)                   
        
        # calculate mu_w given the current velocity profile
        
        for k in range(len(muW)):
            result = minimize(muI.muW_minimize, muW_, (H[k],W[k],U[k]*secsDay),  method='COBYLA', constraints=[muW_constraint], tol=1e-6)#, options={'disp': True})
            muW[k]  = result.x
                
        # constructing matrix Dx = T to solve for velocity        
        T = ((2*H[:-1]-d)*np.diff(H)*dx + 2*muW[:-1]/W[:-1]*H[:-1]**2*dx**2)
        T[0] = Ut # upstream boundary moves at terminus velocity
        T = np.append(T,0)#(1-d/H[-1])*ee[-1]/mu[-1]) # downstream boundary condition
                      
    
        A = nu[:-1]
        B = -(nu[:-1]+nu[1:])
        C = nu[1:]

        # use a_left, a, and a_right define the diagonals of D
        a_left = np.append(A, -1)
        
        a = np.ones(len(T)) # set to positive one because default is to set strain rate equal to zero
        a[1:-1] = B
        a[-1] = 1
                                   
        a_right = np.append(0,C)
        
          
        # print('a: ' + str(len(a)))
        # print('a_left: ' + str(len(a_left)))
        # print('a_right: ' + str(len(a_right)))
        
        diagonals = [a_left,a,a_right]
        D = diags(diagonals,[-1,0,1]).toarray() 
         
        U_new = np.linalg.solve(D,T) # solve for velocity
        
        
        
        if (np.abs(U-U_new)*secsYear > 0.1).any():        
                U = U_new
        else:
            U = U_new
            #plt.plot(x,mu)
            break
               
    return(U, mu)


#%%


dt = secsDay*1

B = 0/secsYear # mass balance rate in m/s

x0 = 0
L = 1e4+x0 # melange length
dx = 100 # grid spacing
x = np.arange(x0,L+dx,dx) # longitudinal grid
W = 5000*np.ones(x.shape) # melange width; treated as constant for now
muW = muS*np.ones(x.shape)


#H = (-200/L*x + 200) + d
#H = -50*(x/L-1) + d # initial melange thickness
H = np.ones(x.shape)*(d) # melange thickness

#Ut = 1000/secsYear # glacier terminus velocity
#U = Ut*(1+x/L) #np.ones(len(x)) # initial guess for the velocity

U = np.zeros(x.shape)
U[0] = 10000/secsYear


#%%

ax1 = plt.subplot(311)
ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Speed [m/yr]')
ax1.set_ylim([0,6000])

ax2 = plt.subplot(312)
ax2.set_xlabel('Longitudinal coordinate [m]')
ax2.set_ylabel('Thickness [m]')
ax2.set_ylim([0, 100])

ax3 = plt.subplot(313)
ax3.set_xlabel('Longitudinal coordinate [m]')
ax3.set_ylabel('$\mu$')
ax3.set_ylim([0.1, 0.7])

n = 1 # number of time steps
color_id = np.linspace(0,1,n) 


for k in np.arange(0,n):
         
    U, mu = velocity(x,U,H,W,dx)
             
    
    dHdt = (B-np.gradient(H*W*U,x)/W)
    #dHdt[-1] = 0
    # update thickness
    H += dHdt*dt
    #H[:-1] += (B-np.diff(H*W*U)/W[:-1]/dx)*dt
    
            
    # update grid
    x0 = x[0]+Ut*dt
    xL = x[-1]+U[-1]*dt

    xg = np.linspace(x0,xL,len(x))
    dx = x[1]-x[0]
    
    H_interp1d = interp1d(x, H, kind='linear', fill_value='extrapolate') #, right=d) # linear extrapolation???
    H = H_interp1d(xg)
    
    U_interp1d = interp1d(x, U, kind='linear', fill_value='extrapolate')
    U = U_interp1d(xg)
    
    W = np.interp(xg, x, W)
    x = xg

    if k % 10 == 0:
        ax1.plot(x,U*secsYear,color=plt.cm.viridis(color_id[k]))
        ax2.plot(x,H,color=plt.cm.viridis(color_id[k]))
        ax3.plot(x,mu,color=plt.cm.viridis(color_id[k]))
        
plt.tight_layout()

