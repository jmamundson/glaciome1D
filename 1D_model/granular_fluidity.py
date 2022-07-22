# nonlocal.py contains tools for using the granular fluidity rheology

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags

from config import *

from matplotlib import pyplot as plt

#%% calculate the effective pressure driving flow

# W = ice melange width
# L = ice melange length
# mu0 = coefficient of friction at yield stress
def pressure(H):
    
    P = 0.5*rho*g*(1-rho/rho_w)*H
    
    return(P)

#%%

def calc_mu(x,U,H,dx):
    
    # note: use strain rate to compute mu, which is used to compute g'
    # need to iterate because sign(exx) is not necessarily always the same
    
    # constants --> Needs some thought!
    #b = (mu0-muS)/I0
    #A = 0.5

    dee = 1e-15 # finite strain rate to prevent infinite viscosity
    ee = np.abs(np.gradient(U,dx)/2)+dee
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
     
        #print(np.max(np.abs(gg-gg_new)))
        #print(k)
        if (np.abs(gg-gg_new) > 1e-6).any():        
        #if np.max(np.abs(gg-gg_new)) > 1e-10:
            #print(np.max(np.abs(gg-gg_new)))
            gg = gg_new
            gg[gg==0] = 1e-6 # small factor to make mu real; doesn't do anything because gg=0 when ee=0, so mu=0 in this case
            mu = ee/gg   
        else:
            gg = gg_new
            gg[gg==0] = 1e-6
            mu = ee/gg
            xn = np.linspace(x[0]+dx/2, x[-1]+dx/2, len(x)-1)
            nu = np.interp(xn,x,H**2/gg) # create new variable on staggered grid to simplify later
            
            break
        
    return(nu, mu, ee)
    
    
#%%
def velocity(x,Ut,U,H,W,dx):
    
    # U is the initial guess for the velocity
    # plt.figure(figsize=(10,8))
    # ax1 = plt.subplot(311)
    # ax2 = plt.subplot(312)
    # ax3 = plt.subplot(313)

    muW = muW_*np.ones(x.shape)
    
    j = 1
    while j==1:
        
        #print(U[-1]*secsDay)
        nu, mu, ee = calc_mu(x,U,H,dx)
        #nu, mu = granular_fluidity(x,U,H,dx)                   
        
        # calculate mu_w given the current velocity profile
        
        for k in range(len(muW)):
            result = minimize(muW_minimize, muW_, (H[k],W[k],U[k]),  method='COBYLA', constraints=[nonlocal_constraint], tol=1e-6)#, options={'disp': True})
            muW[k] = result.x
        
        # ax1.plot(x,muW)
        # ax1.set_ylim([muS-0.1, mu0+0.1])
        # ax2.plot(x,mu,'--')
        # ax2.set_ylim([muS-0.1, mu0+0.1])
        # ax3.plot(x,U*secsDay)
        # ax3.set_ylim([-300, 300])
        #plt.semilogy(nu)
        
        
        # I think there is a static vs kinetic friction issue here...
        # if muW = muS => dU/dy = 0 which means that U = 0
        # but that's not what's happening
        
        # constructing matrix Dx = T to solve for velocity        
        T = ((2*H[:-1]-d)*np.diff(H)*dx + 2*muW[:-1]/W[:-1]*H[:-1]**2*np.sign(U[:-1])*dx**2)
        T[0] = Ut # upstream boundary moves at terminus velocity
        T = np.append(T,(1-d/H[-1])*ee[-1]/mu[-1]) # downstream boundary condition
                      
        
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
        
       
        print('dU: ' + str(np.max(np.abs(U-U_new))*secsYear) + ' m/yr')
        
        if (np.abs(U-U_new)*secsYear > 1).any():        
            dU = U-U_new
            U = U - dU
            #U = U_new
        else:
            U = U_new
            break
               
    return(U, mu, muW)
        

#%% solve transverse velocity for nonlocal rheology

# A and b are dimensionless parameters
def transverse(W,muS,muW,H,d,A,b): 
    
    P = pressure(H)
    
    n_pts = 101 # number of points in half-space
    y = np.linspace(0,W/2,n_pts) # location of points
    
    dy = y[1]

    mu = muW*(1-2*y/W)

    y_c = W/2*(1-muS/muW) # critical value of y for which mu is no longer greater 
    # than muS; although flow occurs below this critical value, it is needed for 
    # computing g_loc (below)
    
    zeta = np.sqrt(np.abs(mu-muS))/(A*d)
    
    g_loc = np.zeros(len(y))
    g_loc[y<y_c] = np.sqrt(P/rho)*(mu[y<y_c]-muS)/(mu[y<y_c]*b*d) # local granular fluidity
    
    # first solve for the granular fluidity. we set dg/dy = 0 at
    # y = 0 and at y = W/2    
    
    a = -(2+dy**2*zeta**2)
    a[0] = -1
    a[-1] = 1
    
    a_left = np.ones(len(y)-1)
    a_left[-1] = -1
    
    a_right = np.ones(len(y)-1)
    
    diagonals = [a_left,a,a_right]
    D = sparse.diags(diagonals,[-1,0,1]).toarray()
    
    f = -g_loc*zeta**2*dy**2
    f[0] = 0
    f[-1] = 0
    
    gg = np.linalg.solve(D,f) # solve for granular fluidity
    
    a = np.zeros(len(y))
    a[0] = 1
    a[-1] = 1
    
    a_left = -np.ones(len(y)-1)
    
    a_right = np.ones(len(y)-1)
    a_right[0] = 0
    
    diagonals = [a_left,a,a_right]
    D = sparse.diags(diagonals,[-1,0,1]).toarray()
    
    f = mu*gg*2*dy
    f[0] = 0
    f[-1] = 0
    
    # boundary conditions: u(0) = 0; du/dy = 0 at y = W/2
    
    u = np.linalg.solve(D,f)
    u = u*secsDay
    
    
    u_mean = np.mean(u)
    
    return(y,u,u_mean)


#%%
# needed for determining muW 

def muW_minimize(muW, H, W, U):
    
    _, _, u_mean = transverse(W,muS,muW,H,d,A,b)
    
    # take into account that flow might be in the negative direction
    if U<0:
        u_mean = -u_mean
        
    du = np.abs(U-u_mean)    
    return(du)

#%%
# H = 50
# U = 30
# W = 5000

# k=0
# result = minimize(muW_minimize, muW_, (H,W,U),  method='COBYLA', constraints=[nonlocal_constraint], tol=1e-6)#, options={'disp': True})
# muW = result.x

# y, u, u_mean = transverse(W,muS,muW,H,d,A,b)
# plt.plot(y,u)