# muI.py contains tools for using the mu(I) rheology

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.optimize import minimize

from config import *

from general_utilities import pressure

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

    The minimimization is run during "velocity":
        result = minimize(muW_minimize, muW_, (H[k],W[k],U[k]*secsDay),  method='COBYLA', constraints=[muI_constraint], tol=1e-6)#, options={'disp': True})
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
    
    _, _, u_mean = transverse(H, W, muS, muW, mu0, d, I0)
    
    # take into account that flow might be in the negative direction
    if U<0:
        u_mean = -u_mean
        
    du = np.abs(U-u_mean)    
    return(du)


#%% 
# determine the coefficient of friction in the mu(I) rheology; only accounting for longitudinal strain rates

def calc_mu(x,U,H,dx):
    '''
    Calculates mu for the 1D flow model. In the 1D model, longitudinal and 
    transverse strain rates have been de-coupled. calc_mu only determines mu
    for the longitudinal component, and is primarily used when iterating to 
    the stress balance equation for U.

    Parameters
    ----------
    x : longitudinal position [m]
    U : width-averaged velocity [m s^-1]
    H : ice melange thickness [m]
    dx : grid spacing [m]

    Returns
    -------
    nu : mu*H^2/ee, on the staggered grid
    mu : effective coefficient of friction
    ee : second invariant of the strain rate tensor (neglecting e_xy)
    
    '''
    
    dee = 1e-15 # finite strain rate to prevent infinite viscosity
    # fix later (?); could modify equations so that dU/dx = 0 if ee=0.
    ee = np.sqrt(np.gradient(U,dx)**2/2)+dee # second invariant of strain rate

    I = ee*d/np.sqrt((0.5*g*(1-rho/rho_w)*H))
    mu = muS + I*(mu0-muS)/(I0+I)

    # create staggered grid, using linear interpolation
    xn = np.linspace(x[0]+dx/2, x[-1]+dx/2, len(x)-1 )
    nu = np.interp(xn,x,mu/ee*H**2) # create new variable on staggered grid to simplify later

    return(nu, mu, ee)

    



#%%
def velocity(x,Ut,U,H,W,dx):
    '''
    Primary code for calculating the longitudinal velocity profiles with the
    mu(I) rheology.

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
    U : computed width-averaged velocity [m s^-1]
    mu : effective coefficient of friction
    muW : computed coefficient of friction along the fjord walls

    '''
        
    # plt.figure(figsize=(10,8))
    # ax1 = plt.subplot(311)
    # ax2 = plt.subplot(312)
    # ax3 = plt.subplot(313)

    muW = muW_*np.ones(x.shape) # construct initial array for muW
    
    j = 1
    while j==1:
        
        #print(U[-1]*secsDay)
        nu, mu, ee = calc_mu(x,U,H,dx)
        
        
        # calculate mu_w given the current velocity profile
        for k in range(len(muW)):
            #result = minimize(muW_minimize, muW_, (H[k],W[k],U[k]),  method='COBYLA', constraints=[muI_constraint], tol=1e-10)#, options={'disp': True})
            result = minimize(calc_muW, muW_, (H[k],W[k],U[k]),  method='Nelder-Mead', tol=1e-10)#, options={'disp': True})
            #result = fsolve(calc_muW, muW_, (H[k],W[k],U[k]), xtol=1e-6)#, options={'disp': True})
            if result.x < muS:
                muW[k] = muS
            elif result.x >= mu0:
                muW[k] = mu0-0.0001
            else:
                muW[k]  = result.x
            
                    
        # ax1.plot(x,muW)
        # ax1.set_ylim([muS-0.1, mu0+0.1])
        # ax1.set_ylabel('$\mu_W$')
        # ax2.plot(x,mu)
        # ax2.set_ylim([muS-0.1, mu0+0.1])
        # ax2.set_ylabel('$\mu$')
        # ax3.plot(x,U*secsDay)
        # ax3.set_ylim([-5, 50])
        # ax3.set_ylabel('$U$ [m d$^{-1}$]')
        # ax3.set_xlabel('Longitudinal coordinate [m]')
      
                
        # constructing matrix Dx = T to solve for velocity        
        T = ((2*H[:-1]-d)*np.diff(H)*dx + 2*muW[:-1]/W[:-1]*H[:-1]**2*np.sign(U[:-1])*dx**2)
        T[0] = Ut # upstream boundary moves at terminus velocity
        T = np.append(T,0)#(1-d/H[-1])*ee[-1]/mu[-1]) # downstream boundary condition

        # use a_left, a, and a_right define the diagonals of D
        a_left = np.append(nu[:-1], -1)
        
        a = np.ones(len(T)) # set to positive one because default is to set strain rate equal to zero
        a[1:-1] = -(nu[:-1]+nu[1:])
        a[-1] = 1
                                   
        a_right = np.append(0,nu[1:])
        
        diagonals = [a_left,a,a_right]
        D = diags(diagonals,[-1,0,1]).toarray() 
         
        U_new = np.linalg.solve(D,T) # solve for velocity
               
        if (np.abs(U-U_new)*secsYear > 1).any():        
            dU = U-U_new
            U = U - dU
            #print(np.max(np.abs(dU))*secsDay)
        else:
            U = U_new
            break
               
    return(U, mu, muW)