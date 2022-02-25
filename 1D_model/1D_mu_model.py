import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse import diags

import scipy.integrate as integrate

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
        
        dee = 1e-15 # finite strain rate to prevent infinite viscosity
        # fix later; could modify equations so that dU/dx = 0 if ee=0.
        ee = np.sqrt(np.gradient(U,dx)**2)+dee # second invariant of strain rate
        I = ee*d/np.sqrt((0.5*g*(1-rho/rho_w)*H))
        mu = muS + I*(mu0-muS)/(I0+I)
                           
        # constructing matrix Cx = T to solve for velocity
        # if gg = 0 > modify matrix C so that each line with gg=0 indicates that dU/dx = 0 
        T = np.zeros(len(x))
        T[1:-1] = ((2*H[:-1]-d)*(H[1:]-H[:-1])*dx + 2*mu0/W[:-1]*H[:-1]**2*dx**2)[1:]
        T[0] = Ut # upstream boundary moves at terminus velocity
        T[-1] = 0 # strain rate equals zero at downstream boundary
        
        
        # create staggered grid, using linear interpolation
        #xn = np.linspace(x[0]+dx/2, x[-1]+dx/2, len(x)-1 ) # this needs to be checked!
        xn = np.arange(x[0]+dx/2,x[-1]+dx/2,dx)
        nu = np.interp(xn,x,mu/ee*H**2) # create new variable on staggered grid to simplify later


        a = np.ones(len(T)) # set to positive one because default is to set strain rate equal to zero
        a[0] = 1 # specify upstream boundary condition
        a[1:-1] = -(nu[1:]+nu[0:-1]) # staggered grid
        a[-1] = 1 # needed for specifying downstream boundary condition
        
        a_left = nu
        a_left[-1] = -1 # needed for specifing downstream boundary condition; this line might be redundant
                                
        a_right = nu
        a_right[0] = 0
        
               
        diagonals = [a_left,a,a_right]
        C = diags(diagonals,[-1,0,1]).toarray() 
         
        U_new = np.linalg.solve(C,T) # solve for velocity
        
        #plt.plot(x,mu)
        
        if (np.abs(U-U_new)*secsYear > 1).any():        
                U = U_new
        else:
            U = U_new
            break
               
    return(U)


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
L = 1e4+x0 # melange length
dx = 10 # grid spacing
x = np.arange(x0,L+dx,dx) # longitudinal grid
W = 5000*np.ones(x.shape) # melange width; treated as constant for now

#H = (-200/L*x + 200) + d
H = -200*(x/L-1) + d # initial melange thickness
#H = np.ones(x.shape)*(d) # melange thickness
Ut = 5000/secsYear # glacier terminus velocity


# max and min effective coefficients of friction, sort of
mu0 = 0.6
muS = 0.2
I0 = 10**-7

U = Ut*np.ones(len(x)) # initial guess for the velocity




#%%

ax1 = plt.subplot(211)
ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Speed [m/yr]')
ax2 = plt.subplot(212)
ax2.set_xlabel('Longitudinal coordinate [m]')
ax2.set_ylabel('Thickness [m]')

color_id = np.linspace(0,1,100) 


for k in np.arange(0,100):
         
    U = velocity(x,U,H,W,dx)
    #plt.plot(x,U*secsYear)
    
    if k % 10 == 0:
        ax1.plot(x,U*secsYear,color=plt.cm.viridis(color_id[k]))
        ax2.plot(x,H,color=plt.cm.viridis(color_id[k]))
    
    # update thickness
    H += (B-np.gradient(H*W*U,x)/W)*dt
    
    
    #update grid
    x0 = x[0]+Ut*dt
    xL = x[-1]+U[-1]*dt

    xg = np.linspace(x0,xL,len(x))
    dx = x[1]-x[0]
    H = np.interp(xg, x, H)
    U = np.interp(xg, x, U)
    W = np.interp(xg, x, W)
    x = xg


plt.tight_layout()

