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
def granular_fluidity(x,U,H,dx):
    # note: use strain rate to compute mu, which is used to compute g'
    # need to iterate because sign(exx) is not necessarily always the same
    
    # constants
    b = 2e15
    A = 1

    ee = np.abs(np.gradient(U,dx))
    mu = 0.5*np.ones(len(x)) # initial guess for mu
    gg = ee/mu
    
    j = 1
    while j==1: 
        g_loc = np.zeros(len(x))
        g_loc[mu>muS] = np.sqrt(0.5*g*(1-rho/rho_w)*H[mu>muS]/d**2)*(mu[mu>muS]-muS)/(mu[mu>muS]*b) 
        
        zeta = np.abs(mu-muS)/(A**2*d**2) # zeta = 1/xi^2
    
        # construct equation Cx=T
        # boundary conditions: 
        #    ####g=0 at x=L (implies strain rate = 0)
        #    dg/dx=0 at x=0,L (implies strain rate gradient equals 0)
        
        c_left = np.ones(len(x)-1)
        #c_left[-1] = 0
        
        c = -(2+zeta*dx**2)
        c[0] = -1
        c[-1] = -1
            
        c_right = np.ones(len(x)-1)
            
        diagonals = [c_left,c,c_right]
        C = diags(diagonals,[-1,0,1]).toarray() 
        
        T = -g_loc*zeta*dx**2
        T[0] = 0
        T[-1] = 0
            
        gg_new = np.linalg.solve(C,T) # solve for granular fluidity
     
        
        if (np.abs(gg-gg_new) > 1e-5).any():        
            print(np.max(np.abs(gg-gg_new)))
            gg = gg_new
            #gg[gg==0] = 1e-10 # small factor to make mu real; doesn't do anything because gg=0 when ee=0, so mu=0 in this case
            mu = ee/gg
            
        else:
            gg = gg_new
            break
    #ee = -mu*gg # second invariant of strain rate tensor!
        
    
    return(gg)


#%%
def velocity(x,U,H,W,dx):
    # construct Cx=T to solve for velocity
    #
    #U = vel
    j = 1
    while j==1:
        
        gg = granular_fluidity(x,U,H,dx)
        #plt.plot(x,gg)
        
        ind = (gg>0) # note: granular fluidity can't be less than zero!
        ind = np.sum(ind)
 
        # problem setting up equations if ind is same length as grid due to fact that we use a staggered grid
        # correct here by reducing ind by 1. this is okay because we manually set the last grid points anyway
        if ind == len(gg):
            ind -= 1
            
            
        # currently assumes that strain rate only goes to zero in downstream region!!!
            
        # constructing matrix Cx = T to solve for velocity
        # if gg = 0 > modify matrix C so that each line with gg=0 indicates that dU/dx = 0 
        T = np.zeros(len(x))
        T[1:ind-1] = ((2*H[:-1]-d)*(H[1:]-H[:-1])*dx + 2*muW/W[:-1]*H[:-1]**2*dx**2)[1:ind-1]
        T[0] = Ut # upstream boundary moves at terminus velocity
        T[-1] = (0.5-d/H[-1])*gg[-1]*dx # strain rate equals zero at downstream boundary
        #T[T<0] = 0 # TEMPORARY FIX --> PROBLEM WHEN mu0 IS TOO LARGE???
        
        # create staggered grid, using linear interpolation
        xn = np.arange(dx/2,x[-1]+dx/2,dx)
        Hn = np.interp(xn,x,H)
        ggn = np.interp(xn,x,gg)
        
        nu = Hn**2/ggn # for simplicity, create new variable describing H^2/g' on the staggered grid
  
        a = np.ones(len(T)) # set to positive one because default is to set strain rate equal to zero
        a[0] = 1 # specify upstream boundary condition
        #print(len(nu[1:ind-1]))
        #print(len(nu[0:ind-2]))
        a[1:ind-1] = -(nu[1:ind-1]+nu[0:ind-2]) # staggered grid
        a[1:ind-1] = -2*nu[1:ind-1] # uncomment if you don't want to use a staggered grid
        a[-1] = 1 # needed for specifying downstream boundary condition
        
        a_left = -np.ones(len(x)-1) # set to negative one because default is to set strain rate equal to zero
        a_left[0:ind-2] = nu[0:ind-2]
        a_left[-1] = -1 # needed for specifing downstream boundary condition; this line might be redundant
                                
        a_right = np.zeros(len(a_left)) 
        a_right[0] = 0
        a_right[1:ind-1] = nu[2:ind]
               
        diagonals = [a_left,a,a_right]
        C = diags(diagonals,[-1,0,1]).toarray() 
         
        U_new = np.linalg.solve(C,T) # solve for velocity
        
        if (np.abs(U-U_new)*secsYear > 1).any():        
                U = U_new
        else:
            U = U_new
            break
        
    return(U)


#%%
secsDay = 86400
secsYear = secsDay*365.25
dt = secsDay*0.1

rho = 917 # density of ice
rho_w = 1028 # density of water

g = 9.81 # gravity

B = -10/secsYear # mass balance rate in m/s

d = 25 # characteristic iceberg size

L = 1e4 # melange length
dx = 100 # grid spacing
x = np.arange(0,L+dx,dx) # longitudinal grid
W = 5000*np.ones(x.shape) # melange width; treated as constant for now

#H = (-200/L*x + 200) + d
H = -200*(x/L-1) + d # initial melange thickness
#H = np.ones(x.shape)*(d) # melange thickness
Ut = 5000/secsYear # glacier terminus velocity


# max and min effective coefficients of friction, sort of
muW = 0.6
muS = 0.2


#U = np.zeros(len(x))
#U[0:5] = Ut


U=Ut*(1-x/L)




#%%

ax1 = plt.subplot(211)
ax1.set_xlabel('Longitudinal coordinate [m]')
ax1.set_ylabel('Speed [m/yr]')
ax2 = plt.subplot(212)
ax2.set_xlabel('Longitudinal coordinate [m]')
ax2.set_ylabel('Thickness [m]')

k = 1
for k in np.arange(0,1000):
    
    ax1.plot(x,U*secsYear)
    ax2.plot(x,H)
    
    U = velocity(x,U,H,W,dx)
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


#ee = np.abs(np.gradient(U,x))

#mu = ee/gg
#gg = ee/mu # gg = granular fluidity; ee = second invariant of strain rate tensor



