import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.optimize import minimize
from scipy.optimize import root
from scipy.optimize import newton_krylov

import config

from general_utilities import pressure
from general_utilities import second_invariant

from matplotlib import pyplot as plt
from scipy.integrate import quad

import sys
#%%
def diagnostic(Ugg,x,X,Ut,H,W,dx,L):
    '''
    Used to simultaneously calculate the velocity and the granular fluidity.

    Parameters
    ----------
    Ugg : Array that contains the current value of U and gg.
    x : grid, in transformed coordinate system
    X : grid [m]
    Ut : glacier terminus velocity [m/yr]
    H : ice melange thickness on the staggered grid [m]
    W : fjord width on the staggered grid [m]
    dx : grid spacing, in transformed coordinate system
    L : ice melange length [m]

    Returns
    -------
    res : residual of differential equations

    '''
    
    # extract current values of U and gg
    U = Ugg[:len(x)]
    gg = Ugg[len(x):]
    
    # compute new values of U and gg and determine residual between new and old
    # values
    
    resU = calc_U(U, gg, x, X, Ut, H, W, dx, L)
    resgg = calc_gg(gg, U, H, L, dx) #*L multiply by L to give it similar scale to resU...?
    
    # combine the residuals into one array; this is the array that is being
    # minimized
    res = np.concatenate((resU,resgg))
    
    
    return(res)

#%%
def implicit(UggHL,x,X,Ut,H,W,dx,dt,U_prev,H_prev,L_prev,B):
    '''
    Used to simultaneously calculate the velocity and the granular fluidity.

    Parameters
    ----------
    Ugg : Array that contains the current value of U and gg.
    x : grid, in transformed coordinate system
    X : grid [m]
    Ut : glacier terminus velocity [m/yr]
    H : ice melange thickness on the staggered grid [m]
    W : fjord width on the staggered grid [m]
    dx : grid spacing, in transformed coordinate system
    L : ice melange length [m]

    Returns
    -------
    res : difference between current and new values of velocity and granular 
    fluidity

    '''
    
    # combine the residuals into one array; this is the array that is being
    # minimized
    
    U = UggHL[:len(x)]
    gg = UggHL[len(x):2*len(x)-1]
    H = UggHL[2*len(x)-1:-1]
    L = UggHL[-1]
    
    # residuals for velocity, granular fluidity, thickness, and length
    resU = calc_U(U, gg, x, X, Ut, H, W, dx, L)
    resgg = calc_gg(gg, U, H, L, dx)
    resH = calc_H(H, x, dx, dt, U, U_prev, H_prev, W, L, B)
    resL = (L-L_prev)/dt + U[0] - U[-1]
    resL = resL/L
    
    # append residuals
    resUggHL = np.concatenate((resU, resgg, resH, [resL]))    
    
    
    #print('max res: ' + str(np.max(np.abs(resUggHL))))
    
    return(resUggHL)



#%%
def calc_U(U,gg,x,X,Ut,H,W,dx,L):
    '''
    Calculates the longitudinal velocity profile, which depends on the current
    velocity and ice thickness. The velocity must therefore be calculated
    iteratively. During each iteration, muW (the effective coefficient of
    friction along the fjord walls) is determined, as is gg (the granular
    fluidity). Thus, calc_U makes calls to calc_muW and calc_gg.

    Parameters
    ----------
    U : ice melange velocity, to be determined iteratively [m/yr]
    gg : granular fluidity, from previous iteration [yr^-1]
    x : grid, in transformed coordinate system 
    X : grid [m]
    Ut : glacier terminus velocity [m/yr]; LATER NEED TO ADJUST FOR CALVING
    H : ice melange thickness on the staggered grid [m]
    W : fjord width, on the staggered grid [m]
    dx : grid spacing, in transformed coordinate system
    L : ice melange length [m]
    
    Returns
    -------
    res : residual of the stress balance differential equation

    '''
    
    nu = H**2/gg # for simplicity, later
    #nu = (H-config.d)**2/gg # for simplicity, later
    
    # determine H and W on the grid in order to calculate the coefficient of friction along the fjord walls
    # note that these start at the second grid point
    H_ = (H[:-1]+H[1:])/2 # thickness on the grid, excluding the first and last grid points
    W_ = (W[:-1]+W[1:])/2 # width on the grid, excluding the first and last grid points        
    
    muW = config.muW_*np.ones(H_.shape) # construct initial array for muW on the grid points
        
    ## THIS FOR LOOP CAN BE PARALELLIZED
    # calculate muW given the current velocity profile
    for k in range(len(H_)):
        muW[k] = fsolve(calc_muW, config.muW_, (H_[k],W_[k],U[k+1])) # excluding first and last grid points, where we are prescribing boundary conditions
        
    a_left = nu[:-1]/(dx*L)**2    
    a_left = np.append(a_left,-1/(dx*L))
    
    a = np.ones(len(U))/(dx*L)
    a[1:-1] = -(nu[1:] + nu[:-1])/(dx*L)**2
    
    a_right = nu[1:]/(dx*L)**2
    a_right = np.append(0,a_right)
    
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray() 
    
    T = H_*(H[1:]-H[:-1])/(dx*L) + H_**2/W_*muW*np.sign(U[1:-1])
     
    # upstream boundary condition; for now just set equal to terminus velocity; doesn't account for calving
    T = np.append(Ut/(dx*L),T)
    
    # downstream boundary condition??
    #T = np.append(T,0.5*gg[-1]*(H[-1]-config.d)/H[-1])
    T = np.append(T,0.5*gg[-1])
    
    res = np.matmul(D,U)-T
    
    return(res)      


#%%

def logistic(x,xc,k):
    h = 1/(1+np.exp(-k*(x-xc)))
    return(h)

def calc_df(mu,muS,k):
    df = muS/np.max([mu,muS])**2 # theoretical df/d(mu)
    df_ = df + (logistic(mu,muS,k)-1)/muS # modified df/d(mu)
    
    return(df_)

def calc_gg(gg,U,H,L,dx):
    '''
    Calculates the granular fluidity for the 1D flow model. Longitudinal and 
    transverse strain rates have been de-coupled. calc_gg only determines gg
    for the longitudinal component, and is primarily used when iterating to 
    solve the stress balance equation for U.

    Parameters
    ----------
    gg : granular fluidity that is being determined iteratively [yr^-1]
    H : ice melange thickness [m]
    L : ice melange length [m]
    dx : grid spacing, in the transformed coordinate system
    
    Returns
    -------
    res : residual of the granular fluidity differential equation
    
    '''
    
    # calculate current values of ee_chi, mu, g_loc, and zeta
    
    ee_chi = second_invariant(U,dx) # second invariant of the strain rate in 
    # transformed coordinate system
    
    mu = (ee_chi/L)/(gg+config.dgg) # effective coefficient of friction
    
    mu = np.abs(mu)
    if np.min(mu)<0:
        sys.exit('mu less than 0!')
    # Equation 18 in Amundson and Burton (2018)


##################
    k = 50 # smoothing factor (small number equals more smoothing?)

    #f = np.zeros(mu.shape)

    #for j in np.arange(0,len(mu)):
    #    f[j],_ = quad(calc_df,1e-3,mu[j],args=(config.muS,k))
   
    f = [quad(calc_df,1e-4,x,args=(config.muS,k))[0] for x in mu] 
    f = np.abs(f)
    if np.min(f)<0:
        sys.exit('g_loc less than 0!')
    
    g_loc = config.secsYear*np.sqrt(pressure(H)/config.rho)*f/(config.b*config.d)
    
###################
    #g_loc = config.secsYear*np.sqrt(pressure(H)/config.rho)*(mu-config.muS)/(mu*config.b*config.d)
    #g_loc[g_loc<0] = 0
    
    # approximation of abs(mu-muS)
    f_mu = 2/k*np.log(1+np.exp(k*(mu-config.muS)))-mu+config.muS-2/k*np.log(1+np.exp(-k*config.muS))

    zeta = f_mu/(config.A**2*config.d**2)
    # Essentially Equation 19 in Amundson and Burton (2018)
    #zeta = np.abs(mu-config.muS)/(config.A**2*config.d**2) # zeta = 1/xi^2

    # construct equation Dx=T
    # boundary conditions:
    #    dg/dx=0 is the soft boundary condition recommended by Henann and Kamrin (2013)
    
    bc = 'second-order' # specify whether boundary condition should be 'first-order' accurate or 'second-order' accurate
    # stability issues when using second-order---why???
    
    a_left = np.ones(len(mu)-1)/(L*dx)**2
    a = -(2/(L*dx)**2 + zeta)    
    a_right = np.ones(len(mu)-1)/(L*dx)**2 
       
    if bc=='second-order':
        a_left[-1] = 2/(L*dx)**2
        a_right[0] = 2/(L*dx)**2
    
    elif bc=='first-order':
        a_left[-1] = -1/(L*dx)        
        a[0] = -1/(L*dx) 
        a[-1] = 1/(L*dx) 
        a_right[0] = 1/(L*dx)
    
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray() 
    
    T = -zeta*g_loc
    
    if bc=='first-order':
        T[0] = 0 
        T[-1] = 0
    
    res = np.matmul(D,gg) - T

    # doing it this way so that we can make sure that gg is always positive
    #gg_new = np.linalg.solve(D,T)
    #res = gg-gg_new
    
    return(res)


#%%
# needed for determining muW 
def calc_muW(muW, H, W, U):
    '''
    Compares the velocity from the longitudinal flow model to the width-averaged
    velocity from the transverse profile, for given coefficients of friction
    and geometry. For low velocities, there is relatively little deformation and
    muW will be small. For high velocities, muW will be high. There is no slip
    along the fjord walls since the model doesn't assume an upper end to muW.

    The minimimization is run during "calc_U".

    Parameters
    ----------
    muW : effective coefficient of friction along the fjord walls
    H : ice melange thickness [m]
    W : fjord width [m]
    U : width-averaged velocity from the longitudinal flow model [m yr^-1]

    Returns
    -------
    du : the difference between the width-averaged velocity from the flow model 
    and the width-averaged velocity from the transverse velocity profile

    '''
    
    _, _, u_mean = transverse(W,muW,H)
    
    # take into account that flow might be in the negative direction
    if U<0:
        u_mean = -u_mean
        
    du = np.abs(U-u_mean)    
    
    return(du)

#%%    
def calc_H(H,x,dx,dt,U,U_prev,H_prev,W,L,Bdot):    
    '''
    Calculates the ice melange thickness profile, using an implicit time step. 
    The thickness depends on the current velocity as well as the velocity and
    thickness from the previous time step.

    Parameters
    ----------
    x : grid, in transformed coordinate system
    dx : grid spacing, in transformed coordinate system
        DESCRIPTION.
    dt : time step [yr]
    U : ice melange velocity [m/yr]
    U_prev : ice melange velocity from previous time step [m/yr]
    H_prev : ice melange thickness from previous time step, on the staggered grid [m]
    W : fjord width, on the staggered grid [m]
    L : ice melange length [m]; needs to be determined using an implicit time step [m]
    Bdot : surface + basal mass balance rate [m/yr]; can be specified as a scalar or a vector on the staggered grid

    Returns
    -------
    H_new : thickness for the next iteration; H must be adjusted iteratively 
    until H_new= H

    '''
    
    xs = (x[:-1]+x[1:])/2 # staggered grid in the transformed coordinate system
    
    # is this expression for beta wrong???
    #beta = U[0]+xs*(U[-1]-U[0]-U_prev[-1]+U_prev[0])
    beta = U[0]+xs*(U[-1]-U[0])
          
                             
    a_left = dt/(2*dx*L)*(beta[1:] - W[:-1]/W[1:]*(U[1:-1]+U[:-2]))
    a_left[-1] = dt/(dx*L)*(beta[-1]-0.5*W[-2]/W[-1]*(U[-2]+U[-3]))
    
    a = 1 + dt/(2*dx*L)*(U[2:]+U[1:-1])
    a[-1] = 1+dt/(dx*L)*(-beta[-1]+0.5*(U[-1]+U[-2])) # check sign in front of beta!
    a = np.append(1+beta[0]*dt/(L*dx)-dt/(2*L*dx)*(U[1]+U[0]), a)
    
    a[0] = 1 + dt/(dx*L)*(beta[0] - (U[0]+U[1])/2) 
    
    a_right = -dt/(2*dx*L)*beta[1:]
    a_right[0] = dt/(L*dx)*(-beta[0]+(U[2]+U[1])/2*W[1]/W[0])
    
    T = Bdot*dt + H_prev
    
    # attempt at setting d^2{H}/dx^2=0 
    a[0] = 1
    a_right[0] = -2
    
    
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray()
    
    # attempt at setting d^2{H}/dx^2=0
    D[0,2] = 1
    T[0] = 0
    
    res = np.matmul(D,H) - T

    return(res)


#%% solve transverse velocity for nonlocal rheology
def transverse(W,muW,H): 
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
    
    #P = pressure(H)
    
    n_pts = 101 # number of points in half-space
    y = np.linspace(0,W/2,n_pts) # location of points
    
    dy = y[1]

    mu = muW*(1-2*y/W)

    y_c = W/2*(1-config.muS/muW) # critical value of y for which mu is no longer greater 
    # than muS; although flow occurs below this critical value, it is needed for 
    # computing g_loc (below)
       
    zeta = np.sqrt(np.abs(mu-config.muS))/(config.A*config.d)
    
    g_loc = np.zeros(len(y))
    g_loc[y<y_c] = config.secsYear*np.sqrt(pressure(H)/config.rho)*(mu[y<y_c]-config.muS)/(mu[y<y_c]*config.b*config.d) # local granular fluidity
    
    
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
def get_muW(x,U,H,W,L,dx):
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
    
    # ee_chi = second_invariant(U,dx) # second invariant of the strain rate in 
    # # transformed coordinate system
    
    # gg = config.secsYear*1e-7*np.ones(len(x)-1) # initial guess for the granular fluidity
    
    # result = root(calc_gg, gg, (ee_chi,H,L,dx), method='hybr', tol=1e-6)
    # gg = result.x
    # #gg = fsolve(calc_gg, gg, (ee_chi,H,L,dx))
    
    
    # mu = (ee_chi/L)/gg
    
    H_ = (H[:-1]+H[1:])/2
    W_ = (W[:-1]+W[1:])/2
    muW = config.muW_*np.ones(H_.shape)
    
    # calculate mu_w given the current velocity profile
    for k in range(len(H_)):
        result = root(calc_muW, config.muW_, (H_[k],W_[k],U[k+1]), method='lm', options={'xtol':1e-6})
        muW[k] = result.x
        #result = fsolve(calc_muW, config.muW_, (H_[k],W_[k],U[k+1]))
        #muW[k] = result
        #muW[k] = result.x
        
        
    return(muW)

#%%
