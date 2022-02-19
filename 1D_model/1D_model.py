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
    b = 2e5
    A = 0.5

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
        #    g=0 at x=L (implies strain rate = 0)
        #    dg/dx=0 at x=0 (implies strain rate gradient equals 0)
        
        c_left = np.ones(len(x)-1)
        c_left[-1] = 0
        
        c = -(2+zeta*dx**2)
        c[0] = -1
        c[-1] = 1
            
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
# # calculate mu by integrating (alternative appraoch to 1D equations)
# def calc_mu(x,H,W,dx):
       
#     integrand = (2*H-d)*np.gradient(H,x)+2*(mu0/W)*H**2
    
#     integral = -np.append(0,integrate.cumtrapz(integrand[::-1], dx=dx))[::-1]
    
#     mu = -np.sign(-1)/H**2*integral # for now, not accounting for actual sign of exx!!!!
    
#     return(mu)

# #%%
# def calc_gg(x,H,mu):
    
#     b = 2e8
#     A = 1
    
#     g_loc = np.zeros(len(x))
#     g_loc[mu>muS] = np.sqrt(0.5*g*(1-rho/rho_w)*H[mu>muS]/d**2)*(mu[mu>muS]-muS)/(mu[mu>muS]*b) 
        
        
#     zeta = np.abs(mu-muS)/(A**2*d**2) #1/xi^2
    
#     # construct equation Cx=T
#     # boundary conditions: 
#     #    g=0 at x=L (implies strain rate = 0)
#     #    dg/dx=0 at x=0 (implies strain rate gradient equals 0)
        
#     c_left = np.ones(len(x)-1)
#     c_left[-1] = 0
    
#     c = -(2+zeta*dx**2)
#     c[0] = -1
#     c[-1] = 1
        
#     c_right = np.ones(len(x)-1)
        
#     diagonals = [c_left,c,c_right]
#     C = diags(diagonals,[-1,0,1]).toarray() 
    
#     T = -g_loc*zeta*dx**2
#     T[0] = 0
#     T[-1] = 0
            
#     gg = np.linalg.solve(C,T) # solve for granular fluidity
    
#     return(gg)
    
# #%%
# mu = calc_mu(x,H,W,dx)
# gg = calc_gg(x,H,mu)

# ee = mu*gg
# exx = -ee

# U = (Ut+np.append(0,integrate.cumtrapz(exx,dx=dx)))
# plt.plot(x,U*secsYear)

# H += (B - np.gradient(U*H*W,dx)/W)*dt

# use integral to solve directly for 1-D case...
    #integrand = (2*H-d)*np.gradient(H,x)+2*(mu0/W)*H**2

    # int_x^L f(x) dx = -int_L^x f(x) dx; need to reverse the order twice to do this numerically...
    #integral = np.append(0,integrate.cumtrapz(-integrand[::-1], dx=dx))
    #integral = integral[::-1]

    #mu = np.sign(exx)/H**2*integral
#%%
secsDay = 86400
secsYear = secsDay*365.25
dt = secsDay*0.5

rho = 917 # density of ice
rho_w = 1028 # density of water

g = 9.81 # gravity

B = -10/secsYear # mass balance rate in m/s

d = 25 # characteristic iceberg size

L = 1e4 # melange length
dx = 10 # grid spacing
x = np.arange(0,L+dx,dx) # longitudinal grid
W = 5000*np.ones(x.shape) # melange width; treated as constant for now

#H = (-200/L*x + 200) + d
H = -200*(x/L-1) + d # initial melange thickness
#H = np.ones(x.shape)*(d) # melange thickness
Ut = 5000/secsYear # glacier terminus velocity


# max and min effective coefficients of friction, sort of
mu0 = 0.5
muS = 0.2


#U = np.zeros(len(x))
#U[0:5] = Ut


U=Ut*(1-x/L)


#%%


def velocity(x,U,H,dx):
    # construct Cx=T to solve for velocity
    #
    #U = vel
    j = 1
    while j==1:
        
        gg = granular_fluidity(x,U,H,dx)
        #plt.plot(x,gg)
        
        ind = (gg>0) # note: granular fluidity can't be less than zero!
        ind = np.sum(ind)
        # currently assumes that strain rate only goes to zero in downstream region!!!
            
        # constructing matrix Cx = T to solve for velocity
        # if gg = 0 > modify matrix C so that each line with gg=0 indicates that dU/dx = 0 
        T = np.zeros(len(x))
        T[1:ind-1] = ((2*H[:-1]-d)*(H[1:]-H[:-1])*dx + 2*mu0/W[:-1]*H[:-1]**2*dx**2)[1:ind-1]
        T[0] = Ut # upstream boundary moves at terminus velocity
        T[-1] = 0 # strain rate equals zero at downstream boundary
        #T[T<0] = 0 # TEMPORARY FIX --> PROBLEM WHEN mu0 IS TOO LARGE???
        
        # create staggered grid, using linear interpolation
        xn = np.arange(dx/2,L+dx/2,dx)
        Hn = np.interp(xn,x,H)
        ggn = np.interp(xn,x,gg)
        
        nu = Hn**2/ggn # for simplicity, create new variable describing H^2/g' on the staggered grid
        
        a = np.ones(len(T)) # set to positive one because default is to set strain rate equal to zero
        a[0] = 1 # specify upstream boundary condition
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

k = 1

for k in np.arange(0,500):
    plt.subplot(211)
    plt.plot(x,U*secsYear)
    plt.subplot(212)
    plt.plot(x,H)
    
    U = velocity(x,U,H,dx)
    # update thickness
    H += (B-np.gradient(H*W*U,x)/W)*dt
    
    
    # update grid
    x0 = x[0]#+Ut*dt
    xL = x[-1]+U[-1]*dt

    xg = np.linspace(x0,xL,len(x))
    dx = x[1]-x[0]
    H = np.interp(xg, x, H)
    U = np.interp(xg, x, U)
    x = xg





#ee = np.abs(np.gradient(U,x))

#mu = ee/gg
#gg = ee/mu # gg = granular fluidity; ee = second invariant of strain rate tensor



