# SOMETHING STRANGE GOING ON WITH THE COORDINATE TRANSFORMATION!!!!

# muI.py contains tools for using the mu(I) rheology
# many of these functions can probably be used in the granular fluidity
# if so, move them into general utilities

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.optimize import minimize
from scipy.optimize import root

import config

from general_utilities import pressure
from general_utilities import second_invariant

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
    g_loc[y<y_c] = np.sqrt(P/config.rho)*(mu[y<y_c]-muS)/(mu[y<y_c]*b*d) # local granular fluidity
    
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
    
    _, _, u_mean = transverse(W,config.muS,muW,H,config.d,config.A,config.b)
    
    # take into account that flow might be in the negative direction
    if U<0:
        u_mean = -u_mean
        
    du = np.abs(U-u_mean)    
    
    return(du)


#%%
def calc_gg(gg,ee_chi,H,L,dx):
    '''
    Calculates mu for the 1D flow model. In the 1D model, longitudinal and 
    transverse strain rates have been de-coupled. calc_mu only determines mu
    for the longitudinal component, and is primarily used when iterating to 
    solve the stress balance equation for U.

    Parameters
    ----------
    H : ice melange thickness [m]
    L : ice melange length [m]
    ee_chi : second invariant of the strain rate on the stretched grid

    Returns
    -------
    mu : effective coefficient of friction
    
    '''
    
    ee = ee_chi/L  
             
    mu = ee/gg
    
    # Equation 18 in Amundson and Burton (2018)
    g_loc = np.zeros(len(mu))
    g_loc[mu>config.muS] = np.sqrt(pressure(H[mu>config.muS])/config.rho)*(mu[mu>config.muS]-config.muS)/(mu[mu>config.muS]*config.b*config.d) 
    
    # Essentially Equation 19 in Amundson and Burton (2018)
    zeta = np.abs(mu-config.muS)/(config.A**2*config.d**2) # zeta = 1/xi^2

    # construct equation Cx=T
    # boundary conditions: 
    #    g=0 at x=0, L (implies strain rate = 0??? or infinite???)
    #    dg/dx=0 is the soft boundary condition recommended by Henann and Kamrin (2013)
    
    c_left = np.ones(len(mu)-1)
    c_left[-1] = -1
    
    c = -(2+zeta*(L*dx)**2)
    c[0] = -1
    c[-1] = 1
        
    c_right = np.ones(len(mu)-1) 
    
    diagonals = [c_left,c,c_right]
    C = diags(diagonals,[-1,0,1]).toarray() 
    
    T = -g_loc*zeta*(L*dx)**2
    T[0] = 0
    T[-1] = 0
        
    gg_new = np.linalg.solve(C,T) # solve for granular fluidity
    
    
        
    dgg = gg-gg_new # difference between previous and current iterations
    
    return(dgg)


#%%
def get_mu(x,U,H,W,L,dx):
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
    
    ee_chi = second_invariant(U,dx) # second invariant of the strain rate in 
    # transformed coordinate system
    
    gg = 1e-9*np.ones(len(x)-1) # initial guess for the granular fluidity
    
    gg = root(calc_gg, gg, (ee_chi,H,L,dx), method='lm', options={'xtol':1e-6})
    #print(gg.success)
    gg = gg.x
    
    mu = (ee_chi/L)/gg
    
    H_ = (H[:-1]+H[1:])/2
    W_ = (W[:-1]+W[1:])/2
    muW = config.muW_*np.ones(H_.shape)
    
    # calculate mu_w given the current velocity profile
    for k in range(len(H_)):
        result = root(calc_muW, config.muW_, (H_[k],W_[k],U[k+1]), method='lm', options={'xtol':1e-9})
        muW[k] = result.x
            
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
    
    gg = 1e-7*np.ones(len(x)-1) # initial guess for the granular fluidity
    
    #gg = root(calc_gg, gg, (ee_chi,H,L,dx), method='lm', options={'xtol':1e-6})
    #print(gg.success)
    #gg = gg.x
     
    nu = 1/gg
    
    # determine H and W on the grid in order to calculate the coefficient of friction along the fjord walls
    H_ = (H[:-1]+H[1:])/2 # thickness on the grid
    W_ = (W[:-1]+W[1:])/2 # width on the grid        
    muW = config.muW_*np.ones(H_.shape) # construct initial array for muW on the grid points
        
    ## NOTE: THIS FOR LOOP CAN BE PARALELLIZED
    # calculate mu_w given the current velocity profile
    for k in range(len(H_)):
        result = root(calc_muW, config.muW_, (H_[k],W_[k],U[k+1]), method='lm', options={'xtol':1e-9})
        muW[k] = result.x
    
        
    a_left = nu[:-1]*H[:-1]**2    
    a_left = np.append(a_left,-1)
        
    a = np.ones(len(U))
    a[1:-1] = -(nu[1:]*H[1:]**2 + nu[:-1]*H[:-1]**2)
    
    a_right = nu[1:]*H[1:]**2
    a_right = np.append(0,a_right)
    
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray() 

    T = L*((H[1:]+H[:-1])/2)*(H[1:]-H[:-1])*dx + 0.5*L**2*(H[1:]+H[:-1])**2/(W[1:]+W[:-1])*muW*np.sign(U[1:-1])*dx**2
         
    # upstream boundary condition; for now just set equal to terminus velocity; doesn't account for calving
    T = np.append(Ut,T)
    
    # downstream boundary condition
    T = np.append(T,0.5*gg[-1]*dx) 
     
    U_new = np.linalg.solve(D,T) # solve for new velocity
  
    return(U_new)      


#%%    
def spinup(U,x,X,Ut,H,W,dx,dt):
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
    
    xL = X[-1]+U[-1]*dt
    xt = X[0] + U[0]*dt
    
    L = xL-xt # ice melange length [m]
    
    U_new = velocity(U,x,X,Ut,H,W,dx,L) # ice melange velocity based on previous iteration of U [m/s]
    dU = U-U_new
    
    return(dU)   
