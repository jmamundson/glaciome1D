# muI.py contains tools for using the mu(I) rheology

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve

from config import *


#%% convert effective pressure to ice melange thickness in quasi-static case
def thickness(W,L,H_L,mu0):
    #H = 2*P/(rho*(1-rho/rho_w)*g)
    
    H_x = lambda H,x : H - H_L*np.exp(mu0*(L-x)/W+1/2.-H_L/(2*H))
    
    H_guess = 200

    x = np.linspace(0,L,101)
    H = np.zeros(len(x))
    
    for j in range(0,len(x)):
        H[j] = fsolve(H_x,H_guess,x[j])
    
    return(x,H)

#%% calculate the effective pressure driving flow

# W = ice melange width
# L = ice melange length
# mu0 = coefficient of friction at yield stress
def pressure(H):
    
    P = 0.5*rho*g*(1-rho/rho_w)*H
    
    return(P)

#%% calculate resistive force (a.k.a. buttressing force) against the glacier terminus

# ratio = L/W
def force(H_L,mu0):
    
    H_0 = lambda H,ratio : H - H_L*np.exp(mu0*ratio+1/2.-H_L/(2*H))
    
    ratio = np.linspace(0,8,501)
    H_ratio = np.zeros(len(ratio))
    
    H_guess = 200
    
    for j in range(0,len(ratio)):
        H_ratio[j] = fsolve(H_0,H_guess,ratio[j])
        
        
    F = 0.5*rho*g*(1-rho/rho_w)*H_ratio*(H_ratio-H_L)
    
    return(ratio,F)



#%% solve transverse velocity profile for mu(I) rheology

# NEED TO BE CAREFUL WITH GLOBAL VS LOCAL VARIABLES!!!
# create dictionary to pass variables?

# muS = minimum coefficient of friction
# mu0 = maximum coefficient of friction
# d = grain size
# I0 = dimensionless parameter
# muW = coefficient of friction along the fjord wall (muS <= muW < mu0)

def transverse(H, W, muS, muW, mu0, d, I0):
        
    P = pressure(H)  
    
    y_c = W/2*(1-muS/muW) # critical distance from the fjord walls beyond which no deformation occurs
    
    n_pts = 101 # number of points in half-space
    y = np.linspace(0,y_c,n_pts) # location of points
    
    dy = y[1]
    
    Gamma = -2*I0*np.sqrt(P/rho)/d # leading term in equation of velocity profile
    
    mu = muW*(1-2*y/W)

    # RHS of differential equation
    b = Gamma*(mu-muS)/(mu-mu0) 
    b[0] = 0 # set boundary condition of u=0; later this will be adjusted to ensure mass continuity
    b[-1] = 0 # set boundary condition of du/dy=0
    b = b*2*dy
    
    # centered differences
    a = np.zeros(len(y))
    a[0] = 1
    a[-1] = 1
   
    a_left = -np.ones(len(y)-1)
    
    a_right = np.ones(len(y)-1)
    a_right[0] = 0
    
    diagonals = [a_left,a,a_right]
    A = sparse.diags(diagonals,[-1,0,1]).toarray()
    
    u = np.linalg.solve(A,b)
    u = u*secsDay
    
    u = np.append(u,u[-1])
    y = np.append(y,W/2)
    
    u_mean = np.mean(u)*2*y_c/W + np.max(u)*(1-2*y_c/W)
    
    return(y,u,u_mean)

#%% 
# determine the coefficient of friction in the mu(I) rheology; only accounting for longitudinal strain rates

def mu(x,U,H,dx):

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
# needed for determining muW 

def muW_minimize(muW, H, W, U):
    
    _, _, u_mean = transverse(H, W, muS, muW, mu0, d, I0)
    du = np.abs(U-u_mean)    
    return(du)



#%%

# H = 100
# W = 5000
# U = 75

# result = minimize(muW_minimize, muW_, (H,W,U),  method='COBYLA', constraints=[muW_constraint], tol=1e-6, options={'disp': True})
# muW  = result.x
# print(muW)

# #%%
# y, u, u_mean = transverse(H,W,muS,muW,mu0,d,I0)

# y = y-y[-1]
# y = np.append(y,np.abs(y[::-1]))
# u = np.append(u,u[::-1])
# plt.plot(y,u)

