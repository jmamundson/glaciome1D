# muI.py contains tools for using the mu(I) rheology
# many of these functions can probably be used in the granular fluidity
# if so, move them into general utilities

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.optimize import minimize
from scipy.optimize import root

from config import *

from general_utilities import pressure
from general_utilities import second_invariant

from matplotlib import pyplot as plt

#%% solve transverse velocity profile for mu(I) rheology
def transverse(H, W, muS, muW, mu0, d, I0):
    '''
    Calculates transverse velocity profiles for the mu(I) rheology. See
    Amundson and Burton (2018) for details.

    Parameters
    ----------
    H : ice melange thickness [m]
    W : fjord width [m]
    muS : effective coefficient of friction at the yield stress
    muW : effective coefficient of friction along the fjord walls
    mu0 : maximum effective coefficient of friction
    d : characteristic grain size [m]
    I0 : dimensionless parameter, should be on the order of 10^-6

    Returns
    -------
    y : transverse coordinate [m]
    u : velocity at y, assuming no slip along the boundary [m s^-1]
    u_mean : mean velocity across the profile, assuming no slip along the boundary [m s^-1]

    '''
    
    P = pressure(H)  
    
    y_c = W/2*(1-muS/muW) # critical distance from the fjord walls beyond which no deformation occurs
    
    n_pts = 101 # number of points in half-space
    y = np.linspace(0,y_c,n_pts) # location of points
    
    dy = y[1]
    
    Gamma = -2*I0*np.sqrt(P/rho)/d # leading term in equation of velocity profile
    
    mu = muW*(1-2*y/W)
    mu[mu==mu0] = mu0-0.0001
    
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
    u = u
    
    u = np.append(u,u[-1])
    y = np.append(y,W/2)
    
    u_mean = np.mean(u)*2*y_c/W + np.max(u)*(1-2*y_c/W)
    
    return(y,u,u_mean)

#%%
# needed for determining muW 
def calc_muW(muW, H, W, U):
    '''
    Compares the velocity from the longitudinal flow model to the width-averaged
    velocity from the transverse profile, for given coefficients of friction
    and geometry. For low velocities, there is relatively little deformation and
    muW will be small. For high velocities, muW will be high and there may be 
    slip along the fjord walls.

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
    
    _, _, u_mean = transverse(H, W, muS, muW, mu0, d, I0)
    
    # take into account that flow might be in the negative direction
    if U<0:
        u_mean = -u_mean
        
    du = np.abs(U-u_mean)  
    
    return(du)


#%% 
def calc_mu(ee_chi,H,L):
    '''
    Calculates mu for the 1D flow model. In the 1D model, longitudinal and 
    transverse strain rates have been de-coupled. calc_mu only determines mu
    for the longitudinal component, and is primarily used when iterating to 
    solve the stress balance equation for U. mu is calculated on the grid
    using one-sided differences, meaning that it is evaluated from j=0:N-1.

    Parameters
    ----------
    H : ice melange thickness [m]
    L : ice melange length [m]
    ee_chi : second invariant of the strain rate on the stretched grid

    Returns
    -------
    nu : mu/ee
    mu : effective coefficient of friction
    
    '''
       
    ee = ee_chi/L # second invariant in unstretched coordinates
    I = ee*d/np.sqrt((0.5*g*(1-rho/rho_w)*H)) # inertial number
    mu = muS + I*(mu0-muS)/(I0+I) 
        
    return(mu)

#%%
def get_mu(x,U,H,W,dx,ee_chi,L):
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
    mu = calc_mu(ee_chi,H,L)
    
    H_ = (H[:-1]+H[1:])/2
    W_ = (W[:-1]+W[1:])/2
    muW = muW_*np.ones(H_.shape)
    
    # calculate mu_w given the current velocity profile
    for k in range(len(muW)):
        result = root(calc_muW, muW_, (H_[k],W_[k],U[k+1]), method='hybr', options={'xtol':1e-12})#xtol=1e-12)#, options={'disp': True})
        result = result.x
        
        if result < muS:
            muW[k] = muS
        elif result >= mu0:
            muW[k] = mu0-0.0001
        else:
            muW[k]  = result
            
    return(mu,muW)

#%%
def convergence(UH,x,X,Ut,H,W,dx,dt,U_prev,H_prev,B):
    '''
    Primary code for calculating the longitudinal velocity profiles with the
    mu(I) rheology. Use this with root.

    Parameters
    ----------
    x : longitudinal coordinate [m]
    Ut : width-avergaed terminus velocity [m s^-1]
    U : initial guess of the width-averaged velocity [m s^-1]
    H : ice melange thickness [m]
    W : fjord width, on the staggered grid [m]
    dx : grid spacing [m]
    UH : array that contains U on the grid and H on the staggered grid
    
    Returns
    -------
    dU : difference in width-averaged velocity from one iteration to 
    the next [m s^-1]

    '''
    
    # velocity and thickness are needed in the iterations
    U = UH[:len(x)]
    H = UH[len(x):]
    
    # first use implicit time step to find x_L, x_t, and L
    # xL^{n}  = xL^{n-1}+U_L^{n}*dt
    xL = X[-1] + U[-1]*dt # position of end of ice melange
    xt = X[0] + U[0]*dt # terminus position
    L = xL-xt # ice melange length
    
    # use current values of U and H to solve for U and H
    U_new = velocity(U,x,X,Ut,H,W,dx,L)
    H_new = time_step(x,dx,dt,U,U_prev,H_prev,W,L,B)
    
    UH_new = np.append(U_new,H_new)
        
    dU = UH-UH_new    
    
    return(dU)

#%%    
def time_step(x,dx,dt,U,U_prev,H_prev,W,L,Bdot):    
    '''
    Calculates the ice melange thickness profile, using an implicit time step. 
    The thickness depends on the current velocity as well as the velocity and
    thickness from the previous time step.

    Parameters
    ----------
    x : grid, in transformed coordinate system
    dx : grid spacing, in transformed coordinate system
        DESCRIPTION.
    dt : time step [s]
    U : ice melange velocity [m/s]
    U_prev : ice melange velocity from previous time step [m/s]
    H_prev : ice melange thickness from previous time step, on the staggered grid [m]
    W : fjord width, on the staggered grid [m]
    L : ice melange length [m]; needs to be determined using an implicit time step [m]
    Bdot : surface + basal mass balance rate [m/s]; can be specified as a scalar or a vector on the staggered grid

    Returns
    -------
    H_new : thickness for the next iteration; H must be adjusted iteratively 
    until H_new= H

    '''
    
    xs = (x[:-1]+x[1:])/2 # staggered grid in the transformed coordinate system
    
    beta = U[0]-U_prev[0]+xs*(U[-1]-U[0]-U_prev[-1]+U_prev[0])
                                  
    b_left = dt/(2*dx*L)*(beta[1:] - W[:-1]/W[1:]*(U[1:-1]+U[:-2]))
    b_left[-1] = dt/(dx*L)*(beta[-1]-0.5*W[-2]/W[-1]*(U[-2]+U[-3]))
    
    b = 1 + dt/(2*dx*L)*(U[2:]+U[1:-1])
    b[-1] = 1+dt/(dx*L)*(-beta[-1]+0.5*(U[-1]+U[-2]))
    b = np.append(1+beta[0]*dt/(L*dx)-dt/(2*L*dx)*(U[1]+U[0]), b)
    
    b_right = -dt/(2*dx*L)*(U[0]-U_prev[0]+xs[1:]*(U[-1]-U[0]-U_prev[-1]+U_prev[0]))
    b_right[0] = dt/(L*dx)*(-beta[0]+0.5*W[1]/W[0]*(U[2]+U[1]))
    
    TT = Bdot*dt + H_prev
    
    diagonals = [b_left,b,b_right]
    DD = diags(diagonals,[-1,0,1]).toarray()
    
    H_new = np.linalg.solve(DD,TT)

    return(H_new)
    
    
#%%
def velocity(U,x,X,Ut,H,W,dx,L):
    '''
    Calculates the longitudinal velocity profile, which depends on the current
    velocity and ice thickness. The velocity must therefore be calculated
    iteratively.

    Parameters
    ----------
    U : ice melange velocity, to be determined iteratively [m/s]
    x : grid, in transformed coordinate system 
    X : grid [m]
    Ut : glacier terminus velocity [m/s]; LATER NEED TO ADJUST FOR CALVING
    H : ice melange thickness on the staggered grid [m]
    W : fjord width, on the staggered grid [m]
    dx : grid spacing, in transformed coordinate system
    L : ice melange length [L]; in the initial time step it is determined from 
    the ice melange geometry, but in subsequent time steps it needs to be
    determined using an implicit time step.
    
    Returns
    -------
    U_new : velocity for the next iteration; U must be adjusted iteratively
    until U_new = U

    '''
    
    ee_chi = second_invariant(U,dx) # second invariant of the strain rate in 
    # transformed coordinate system
    
    mu = calc_mu(ee_chi,H,L)
    
    nu = (mu-muS)/ee_chi # this needs to be checked/discussed
    
    # determine H and W on the grid in order to calculate the coefficient of friction along the fjord walls
    H_ = (H[:-1]+H[1:])/2 # thickness on the grid
    W_ = (W[:-1]+W[1:])/2 # width on the grid
        
    muW = muW_*np.ones(H_.shape) # construct initial array for muW on the grid points
        
    ## NOTE: THIS FOR LOOP CAN BE PARALELLIZED
    # calculate mu_w given the current velocity profile
    for k in range(len(H_)):
        result = root(calc_muW, muW_, (H_[k],W_[k],U[k+1]), method='lm', options={'xtol':1e-12})#xtol=1e-12)#, options={'disp': True})
        result = result.x
        
        if result < muS:
            muW[k] = muS
        elif result >= mu0:
            muW[k] = mu0-0.0001
        else:
            muW[k]  = result
        
    a_left = nu[:-1]*H[:-1]**2    
    a_left = np.append(a_left,-1)
        
    a = np.ones(len(U))
    a[1:-1] = -(nu[1:]*H[1:]**2 + nu[:-1]*H[:-1]**2)
    
    a_right = nu[1:]*H[1:]**2
    a_right = np.append(0,a_right)
    
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray() 

    T = (H[1:]+H[:-1]-d)*((H[1:]-H[:-1])*dx+dhh*L*dx**2) + L*(H[1:]+H[:-1])**2/(W[1:]+W[:-1])*muW*np.sign(U[1:-1])*dx**2
    
    # upstream boundary condition; for now just set equal to terminus velocity; doesn't account for calving
    T = np.append(Ut,T)
    
    # downstream boundary condition
    muL = 1/np.sqrt(2)*(1-d/H[-1])+muS # coefficient of friction at the end of the ice melange; in most cases muL = muS
    T = np.append(T,I0/(d*L)*np.sqrt(pressure(2*H[-1])/rho)*(muL-muS)/(mu0-muL)) # set strain rate = 0 at terminus for now   
     
    U_new = np.linalg.solve(D,T) # solve for new velocity
  
    return(U_new)      


#%%    
def spinup(U,x,X,Ut,H,W,dx):
    '''
    Small little function that is used to iteratively determine the initial 
    velocity, given the initial geometry and terminus velocity.

    Parameters
    ----------
    U : ice melange velocity, to be solved for iteratively [m/s]
    x : grid, in transformed coordinate system 
    X : grid [m]
    Ut : glacier terminus velocity [m/s]; LATER NEED TO ADJUST FOR CALVING
    H : ice melange thickness on the staggered grid [m]
    W : fjord width
    dx : grid spacing, in transformed coordinate system

    Returns
    -------
    dU : difference in ice melange velocity from one iteration to the next

    '''
    
    L = X[-1]-X[0] # ice melange length [m]
    
    U_new = velocity(U,x,X,Ut,H,W,dx,L) # ice melange velocity based on previous iteration of U [m/s]
    dU = U-U_new
    
    return(dU)   
