# nonlocal.py contains tools for using the granular fluidity rheology

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.optimize import minimize
from scipy.optimize import root


from config import *

from general_utilities import pressure

from matplotlib import pyplot as plt

#%% solve transverse velocity for nonlocal rheology
def transverse(W,muS,muW,H,d,A,b): 
    '''
    Calculates transverse velocity profiles for the nonlocal granular fluidity
    rheology. See Amundson and Burton (2018) for details.

    Parameters
    ----------
    W : fjord width [m]
    muS : effective coefficient of friction at the yield stress (?)
    muW : effective coefficient of friction along the fjord walls
    H : ice melange thickness [m]
    d : characteristic grain size [m]
    A : dimensionless constant referred to as the nonlocal amplitude
    b : dimensionless constant

    Returns
    -------
    y : transverse coordinate [m]
    u : velocity at y, assuming no slip along the boundary [m s^-1]
    u_mean : mean velocity across the profile, assuming no slip along the boundary [m s^-1]

    '''
    
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
def calc_muW(muW, H, W, U):
    '''
    Compares the velocity from the longitudinal flow model to the width-averaged
    velocity from the transverse profile, for given coefficients of friction
    and geometry. For low velocities, there is relatively little deformation and
    muW will be small. For high velocities, muW will be high. There is no slip
    along the fjord walls since the model doesn't assume an upper end to muW.

    The minimimization is run during "velocity":
        result = minimize(calc_muW, muW_, (H[k],W[k],U[k]*secsDay),  method='COBYLA', constraints=[muI_constraint], tol=1e-6)#, options={'disp': True})
        muW[k]  = result.x

    Parameters
    ----------
    muW : effective coefficient of friction along the fjord walls
    H : ice melange thickness [m]
    W : fjord width [m]
    U : width-averaged velocity from the longitudinal flow model [m s^-1]

    Returns
    -------
    du : the difference between the width-averaged velocity from the flow model 
    and the width-averaged velocity from the transverse velocity profile

    '''
    
    _, _, u_mean = transverse(W,muS,muW,H,d,A,b)
    
    # take into account that flow might be in the negative direction
    if U<0:
        u_mean = -u_mean
        
    du = np.abs(U-u_mean)    
    
    return(du)

#%%
def calc_mu(x,U,H,dx):
    '''
    Calculates mu for the 1D flow model. In the 1D model, longitudinal and 
    transverse strain rates have been de-coupled. calc_mu only determines mu
    for the longitudinal component, and is primarily used when iterating to 
    solve the stress balance equation for U.

    Parameters
    ----------
    x : longitudinal position [m]
    U : width-averaged velocity [m s^-1]
    H : ice melange thickness [m]
    dx : grid spacing [m]

    Returns
    -------
    nu : mu*H/gg, on the staggered grid
    mu : effective coefficient of friction
    ee : second invariant of the strain rate tensor (neglecting e_xy)
    
    '''
    
    # set up the staggered grid
    xn = np.linspace(x[0]+dx/2, x[-1]+dx/2, len(x)-1 )
    Hn = np.interp(xn, x, H) # thickness on the staggered grid
    
    dee = 1e-15 # finite strain rate to prevent infinite viscosity
    
    #ee = np.sqrt(0.5*(np.diff(U)/np.diff(x))**2) + dee # second invariant of the strain rate
  
    ee = np.sqrt(0.5*np.gradient(U, x, edge_order=1)**2) + dee # second invariant of the strain rate
    
    
    mu = 2*muS*np.ones(len(x)) # initial guess for mu
    gg = ee/mu # initial guess for the granular fluidity
    
    k = 1
    while k==1: 
        
        # Equation 18 in Amundson and Burton (2018)
        g_loc = np.zeros(len(x))
        g_loc[mu>muS] = np.sqrt(pressure(H[mu>muS])/(rho*d**2))*(mu[mu>muS]-muS)/(mu[mu>muS]*b) 
        
        # Essentially Equation 19 in Amundson and Burton (2018)
        zeta = np.abs(mu-muS)/(A**2*d**2) # zeta = 1/xi^2
    
        # construct equation Cx=T
        # boundary conditions: 
        #    g=0 at x=0, L (implies strain rate = 0)
        #    dg/dx=0 is the soft boundary condition recommended by Henann and Kamrin (2013)
        
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
        
        dgg = gg_new - gg
        
        gg += + dgg/100
        mu = ee/gg   
        #plt.plot(x,gg)
        
        #gg = gg_new
        #print(np.max(np.abs(gg-gg_new)))
        #print(k)
        if (np.abs(dgg) < 1e-10).any():        
            xn = np.linspace(x[0]+dx/2, x[-1]+dx/2, len(x)-1)
            nu = np.interp(xn,x,H/gg)*np.interp(xn,x,H) # create new variable on staggered grid to simplify later
            
            break
        
    return(nu, mu, ee)
    

#%%
def get_mu(x,U,H,W,dx):
    '''
    After determining the velocity profile with fsolve, go back and retrieve
    the effective coefficients of friction.

    Parameters
    ----------
    x : longitudinal coordinate [m]
    U : initial guess of the width-averaged velocity [m s^-1]
    H : ice melange thickness [m]
    W : fjord width [m]
    dx : grid spacing [m]

    Returns
    -------
    mu : effective coefficient of friction
    muW : effective coefficient of friction along the fjord walls

    '''
    
    # calculate mu given the current velocity profile
    _, mu, _ = calc_mu(x,U,H,dx)
    
    muW = muW_*np.ones(x.shape)
    
    # calculate mu_w given the current velocity profile
    for k in range(len(muW)):
        result = root(calc_muW, muW_, (H[k],W[k],U[k]), method='lm', options={'xtol':1e-6})
        #result = minimize(calc_muW, muW_, (H[k],W[k],U[k]),  method='COBYLA', constraints=[nonlocal_constraint], tol=1e-6)#, options={'disp': True})
        muW[k]  = result.x
            
    return(mu,muW)
    
#%%
def velocity(U,x,Ut,H,W,dx):
    '''
    Primary code for calculating the longitudinal velocity profiles with the
    nonlocal granular fluidity rheology.

    Parameters
    ----------
    x : longitudinal coordinate [m]
    Ut : width-avergaed terminus velocity [m s^-1]
    U : initial guess of the width-averaged velocity [m s^-1]
    H : ice melange thickness [m]
    W : fjord width [m]
    dx : grid spacing [m]

    Returns
    -------
    dU : difference in width-averaged velocity from one fsolve iteration to 
    the next [m s^-1]

    '''
    # U is the initial guess for the velocity
    # plt.figure(figsize=(10,8))
    # ax1 = plt.subplot(311)
    # ax2 = plt.subplot(312)
    # ax3 = plt.subplot(313)

    muW = muW_*np.ones(x.shape)
    
    nu, mu, ee = calc_mu(x,U,H,dx)
    
        
    # calculate mu_w given the current velocity profile
    for k in range(len(muW)):
        
        #muW[k] = fsolve(calc_muW, muW, (H[k],W[k],U[k]), xtol=0.1/secsYear)
        result = root(calc_muW, muW_, (H[k],W[k],U[k]), method='lm', options={'xtol':1e-6})
        #result = minimize(calc_muW, muW_, (H[k],W[k],U[k]),  method='COBYLA', constraints=[nonlocal_constraint], tol=1e-6)#, options={'disp': True})
        muW[k] = result.x
    
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
        
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray() 
     
    U_new = np.linalg.solve(D,T) # solve for velocity
    
    dU = U-U_new

              
    return(dU)
        





