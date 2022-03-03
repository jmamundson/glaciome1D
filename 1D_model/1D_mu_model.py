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
# mu(I) rheology

def muI(x,U,H,dx):

    dee = 1e-15 # finite strain rate to prevent infinite viscosity
    # fix later; could modify equations so that dU/dx = 0 if ee=0.
    ee = np.sqrt(np.gradient(U,dx)**2)+dee # second invariant of strain rate

    I = ee*d/np.sqrt((0.5*g*(1-rho/rho_w)*H))
    mu = muS + I*(mu0-muS)/(I0+I)

           # create staggered grid, using linear interpolation
    xn = np.linspace(x[0]+dx/2, x[-1]+dx/2, len(x)-1 )
    nu = np.interp(xn,x,mu/ee*H**2) # create new variable on staggered grid to simplify later

    return(nu, mu)


#%%

def granular_fluidity(x,U,H,dx):
    
    # note: use strain rate to compute mu, which is used to compute g'
    # need to iterate because sign(exx) is not necessarily always the same
    
    # constants
    b = (mu0-muS)/I0
    A = 0.5

    ee = np.abs(np.gradient(U,dx))
    mu = 0.5*np.ones(len(x)) # initial guess for mu
    gg = ee/mu
    
    k = 1
    while k==1: 
        
        g_loc = np.zeros(len(x))
        g_loc[mu>muS] = np.sqrt(0.5*g*(1-rho/rho_w)*H[mu>muS]/d**2)*(mu[mu>muS]-muS)/(mu[mu>muS]*b) 
        
        zeta = np.abs(mu-muS)/(A**2*d**2) # zeta = 1/xi^2
    
        # construct equation Cx=T
        # boundary conditions: 
        #    # g=0 at x=0, L (implies strain rate = 0)
        #    dg/dx=0 at x=0,L (implies strain rate gradient equals 0)
        
        c_left = np.ones(len(x)-1)
        c_left[-1] = -1
        
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
     
        print(np.max(np.abs(gg-gg_new)))
        print(k)
        #if (np.abs(gg-gg_new) > 1e-10).any():        
        if np.max(np.abs(gg-gg_new)) > 1e-10:
            print(np.max(np.abs(gg-gg_new)))
            gg = gg_new
            gg[gg==0] = 1e-10 # small factor to make mu real; doesn't do anything because gg=0 when ee=0, so mu=0 in this case
            mu = ee/gg
            
        else:
            gg = gg_new
            gg[gg==0] = 1e-10
            mu = ee/gg
            xn = np.linspace(x[0]+dx/2, x[-1]+dx/2, len(x)-1 )
            nu = np.interp(xn,x,H**2/gg) # create new variable on staggered grid to simplify later
            
            break
        
        return(nu, mu)
    
#%%
def velocity(x,U,H,W,dx):
    
    # U is the initial guess for the velocity
    
    j = 1
    while j==1:
        
        nu, mu = muI(x,U,H,dx)
        #nu, mu = granular_fluidity(x,U,H,dx)                   
        # constructing matrix Dx = T to solve for velocity
        
        T = ((2*H[:-1]-d)*np.diff(H)*dx + 2*mu0/W[:-1]*H[:-1]**2*dx**2)
        T[0] = Ut # upstream boundary moves at terminus velocity
        T = np.append(T,0)#(1-d/H[-1])*ee[-1]/mu[-1]) # downsream boundary condition
        
 
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
W = 5000*np.ones(x.shape) # melange width; treated as constant for now

#H = (-200/L*x + 200) + d
H = -200*(x/L-1) + d # initial melange thickness
#H = np.ones(x.shape)*(d) # melange thickness
Ut = 5000/secsYear # glacier terminus velocity


# max and min effective coefficients of friction, sort of
mu0 = 0.6
muS = 0.2
I0 = 10**-6

U = Ut*(1+x/L) #np.ones(len(x)) # initial guess for the velocity




#%%

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

    if k % 100 == 0:
        ax1.plot(x,U*secsYear,color=plt.cm.viridis(color_id[k]))
        ax2.plot(x,H,color=plt.cm.viridis(color_id[k]))
        ax3.plot(x,mu,color=plt.cm.viridis(color_id[k]))
        
plt.tight_layout()

