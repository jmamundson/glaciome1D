import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.optimize import root
from scipy.integrate import quad
from scipy.interpolate import interp1d

import config

from matplotlib import pyplot as plt
import matplotlib

import sys

import pickle

#from general_utilities import second_invariant, width, pressure

import os

#%%
class glaciome:
    '''
    The model class contains model variables, which simplifies passing variables
    into and out of functions, and methods for model spinup and implicit and 
    explicit time steps.
    '''
    
    def __init__(self, n_pts, dt, L, Ut):
        '''
        Initialize an object at t=0.

        Parameters
        ----------
        n_pts : number of grid points
        L : ice melange length [m]
        Ut : terminus velocity [m/a]
        W : ice melange width [m]; for now, best to treat as a constant
        '''
        
        # unitless grid and staggered grid
        self.x = np.linspace(0,1,n_pts)
        self.dx = self.x[1]
        self.x_ = (self.x[:-1]+self.x[1:])/2
        
        # grid and staggered grid [m]
        self.L = L
        self.X = self.x*L
        self.X_ = self.x_*L

        # initial thickness (constant), velocity (exponentially decaying), and
        # granular fluidity (linearly decreasing)
        self.H = config.d*np.ones(len(self.x_))
        self.Ut = Ut
        self.U = self.Ut*np.exp(-self.x/0.5)
        self.gg = 1*(1-self.x_)
        self.muW = config.muW_*np.ones(len(self.H)-1)
        
        
        # set initial values for width and mass balance
        self.update_width()
        self.B = 0 # initialize mass balance rate as 0 [m/a]
        
        # time step and initial time
        self.dt = dt
        self.t = 0 # current model time
        
        

    def spinup(self):
        '''
        Run this if a spinup.pickle file does not already exist. Requires a model
        object to have already been created.
        
        Spinup ultimately involves just solving the diagnostic equations for a
        given geometry. In order to the determine a good starting point for 
        solving the diagnostic equations, the code first alternates between 
        solving for the granular fluidity and solving for the velocity. After
        some convergence, it then solves for both of them simultaneously.
        '''
        
    
        # alternate solving for granular fluidity and velocity in order to get a good starting point
        print('Iterating between granular fluidity and velocity in order to find a good starting point')
        residual = self.Ut*np.ones(self.x.shape) # just some large values for the initial residuals
        
        while np.max(np.abs(residual))>1:
            print('Velocity residual: ' + "{:.2f}".format(np.max(np.abs(residual))) + ' m/a')
            
            # solve for granular fluidity
            result = root(calc_gg, self.gg, (self), method='lm')
            self.gg = result.x
            
            # solve for velocity and compute residual
            result = root(calc_U, self.U, (self), method='lm')
            U_new = result.x
            residual = self.U-U_new
            self.U = U_new
        
        print('Done with initial iterations. Solving diagnostic equation.')
        
        # now simultaneous solve for velocity and granular fluidity
        Ugg = np.concatenate((self.U,self.gg))
        result = root(diagnostic, Ugg, (self), method='lm', tol=1e-6)
        self.U = result.x[:len(self.x)]
        self.gg = result.x[len(self.x):]
        
        # update the model object to include the coefficient of friction along
        # the fjord walls that was also computed when solving for the velocity
        self.update_muW() 
        
        # save spin-up file
        file_name = 'spinup.pickle'
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
            file.close()
            print('Spin-up file successfully saved.')
        
    
    def explicit(self):
        '''
        Use an explicit time step to determine the velocity and changes in geometry. 
        First solves for U and gg given the current geometry, then updates the 
        thickness and length.
        '''
             
        # Solve the diagnostic equations for U and gg; first combine into
        # single variable.
        Ugg = np.concatenate((self.U,self.gg))
        result = root(diagnostic, Ugg, self, method='lm', tol=1e-6)
        self.U = result.x[:len(self.x)]
        self.gg = result.x[len(self.x):]
        
        # Update the thickness at each grid point, accounting for advection
        # and coordinate stretching. See Schoof, 2007.
        advection_term = (self.U[0] + self.x_*(self.U[-1]-self.U[0]))*np.gradient(self.H,self.L*self.dx)
        coordinate_stretching_term = ((self.U[2:]+self.U[1:-1])*self.H[1:]*self.W[1:]-(self.U[1:-1]+self.U[:-2])*self.H[:-1]*self.W[:-1])/(2*self.L*self.dx*self.W[1:])
        coordinate_stretching_term = np.append(2*coordinate_stretching_term[0]-coordinate_stretching_term[1],coordinate_stretching_term) # a bit of a hack
        dHdt = self.B + advection_term - coordinate_stretching_term
        self.H += dHdt*self.dt
        
        # Update the length using the velocities at at the end of the melange.
        self.L += (self.U[-1]-self.U[0])*self.dt
        
        # Update the dimensional grid relative the upstream boundary of the
        # melange. Later this should be changed to allow for advection through 
        # a complex fjord geometry.
        self.X = self.x*self.L
        self.X_ = (self.X[:-1]+self.X[1:])/2
        
        # Update the coefficient of friction along the fjord walls.
        self.update_muW()
        
        # Update the time stored within the model object.
        self.t += self.dt
       
    
    def implicit(self):
        '''
        Use an implicit time step to determine the velocity, granular fluidity, 
        thickness, and length.
        '''
        
        # The previous thickness and length are required.
        H_prev = self.H
        L_prev = self.L
        
        UggHL = np.concatenate((self.U,self.gg,self.H,[self.L]))
        result = root(solve_implicit, UggHL, (self, H_prev, L_prev), method='lm', tol=1e-6)
        self.U = result.x[:len(self.x)]
        self.gg = result.x[len(self.x):2*len(self.x)-1]
        self.H = result.x[2*len(self.x)-1:-1]
        self.L = result.x[-1]
        
        # Update the dimensional grid relative the upstream boundary of the
        # melange. Later this should be changed to allow for advection through 
        # a complex fjord geometry.
        self.X = self.x*self.L
        self.X_ = (self.X[:-1]+self.X[1:])/2
        
        # Update the coefficient of friction along the fjord walls.
        self.update_muW()
        
        # Update the time stored within the model object.
        self.t += self.dt
         
        
    def update_muW(self):
        '''
        The coefficient of friction along the fjord walls is calculated when
        iteratively solving for the velocity, but at that point is not directly
        saved in the model object. Here is calculated and stored in the model 
        object by using the current values of the velocity, thickness, and width.
        '''
        
        H_ = (self.H[:-1]+self.H[1:])/2
        W_ = (self.W[:-1]+self.W[1:])/2
        for k in range(len(self.H)-1):
            self.muW[k] = fsolve(calc_muW, config.muW_, (H_[k],W_[k],self.U[k+1])) # excluding first and last grid points, where we are prescribing boundary conditions
          
    def update_width(self):  
        '''
        Update the melange width by interpolating the fjord walls at the grid
        points, X. Currently only set up to handle constant width. 
        '''       
        
        self.W = 4000*np.ones(len(self.X_))
          
          
          
    def save(self,k):
        '''
        Save the model output at time step k.
        '''
        if not os.path.exists('output'): 
            os.mkdir('output')
        
        file_name = 'output/output_' + str(k).zfill(3) + '.pickle'
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
            file.close()
    
    
    
#%%          
def solve_implicit(UggHL, data, H_prev, L_prev):
    '''
    Computes the residual for the velocity, granular fluidity, thickness, and
    length differential equations. The residual is minimized in model.implicit.
    '''
    
    # extract current value for U, gg, H, L
    data.U = UggHL[:len(data.x)]
    data.gg = UggHL[len(data.x):2*len(data.x)-1]
    data.H = UggHL[2*len(data.x)-1:-1]
    data.L = UggHL[-1]

    # calculate residuals
    resU = calc_U(data.U, data)
    resgg = calc_gg(data.gg, data)
    resH = calc_H(data.H, data, H_prev)
    
    # attempt at accounting for calving from the end of the melange; not working!
    # find where thickness first drops below the threshold 
    # H_c = 20   
    # ind = np.where(data.H<H_c)[0]
    # if len(ind)!=0:
    #     H_interp = interp1d(data.H[:ind[0]+1],data.X_[:ind[0]+1])
    #     X_new = H_interp(H_c)
    #     U_c = data.U[-1] - (X_new-L_prev)/data.dt
    # else:
    #     U_c = 0
    
    resL = (data.L-L_prev)/data.dt + data.U[0] - data.U[-1] #+ U_c
    
    # append residuals into single variable to minimized
    resUggHL = np.concatenate((resU, resgg, resH, [resL]))    
    
    
    return(resUggHL)




#%%
def diagnostic(Ugg,data):
    '''
    Used to simultaneously calculate the velocity and the granular fluidity.

    Parameters
    ----------
    Ugg : initial guess for U and gg.
    data : current state of model object

    Returns
    -------
    res : residual of differential equations

    '''
    
        
    # extract current values of U and gg
    data.U = Ugg[:len(data.x)]
    data.gg = Ugg[len(data.x):]
    
    # compute residuals of velocity and granular fluidity differential equations
    resU = calc_U(data.U, data)
    resgg = calc_gg(data.gg, data) 
    
    # combine the residuals into single variable to be minimized
    res = np.concatenate((resU,resgg))
        
    return(res)


#%%
def calc_U(U, data):
    '''
    Compute the residual for velocity differential equation using a fixed geometry
    and granular fluidity.
    
    Parameters
    ----------
    U : initial guess for U
    data : current state of the model object
    
    Returns
    -------
    res : residual of the velocity differential equation

    '''
    
    # extract variables from the model object
    H = data.H
    gg = data.gg
    W = data.W
    dx = data.dx
    L = data.L
    Ut = data.Ut
    muW = data.muW
    
    nu = H**2/gg # for simplicity, later
        
    # determine H and W on the grid in order to calculate the coefficient of friction along the fjord walls
    # note that these start at the second grid point
    H_ = (H[:-1]+H[1:])/2 # thickness on the grid, excluding the first and last grid points
    W_ = (W[:-1]+W[1:])/2 # width on the grid, excluding the first and last grid points        
    
    
    ## THIS FOR LOOP CAN BE PARALELLIZED LATER
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
    
    # downstream boundary condition; longitudinal resistive stress equals
    # difference between glaciostatic and hydrostatic stresses
    T = np.append(T,0.5*gg[-1])
    
    res = np.matmul(D,U)-T
    
    return(res)      


#%%

def logistic(x,xc,k):
    '''
    Returns a logistic function. Used for regularization of g_loc and abs(mu-muS)
    
    Parameters
    ----------
    x : dimensionless grid
    xc : location where step change should occur
    k : amount of smoothing
    
    Returns
    -------
    h : logistic function
    '''
    
    h = 1/(1+np.exp(-k*(x-xc)))
    return(h)


def calc_df(mu,muS,k):
    '''
    f(mu) is used when calculating g_loc. This function returns an approximation
    to the derivative of f(mu).
    
    f(mu) = 1-muS/mu
    df/d(mu) = muS/mu^2

    Parameters
    ----------
    mu : effective coefficient of friction
    muS : effective coefficient of friction at the yield stress
    k : smoothing factor

    Returns
    -------
    df_ : approximation to df/d(mu)

    '''
    df = muS/np.max([mu,muS])**2 # theoretical df/d(mu); max value is to take
    # into account what happens when mu<muS
    df_ = df + (logistic(mu,muS,k)-1)/muS # modified df/d(mu)
    return(df_)


def calc_gg(gg, data):
    '''
    Compute the residual for the granular fluidity differential equation for
    a given geometry and velocity.
    
    Parameters
    ----------
    gg : initial guess for the granular fluidity
    data : current state of the model object
    
    Returns
    -------
    res : residual of the granular fluidity differential equation
    '''
    
    # extract variables from the model object
    U = data.U
    H = data.H
    L = data.L
    dx = data.dx
    
    # calculate current values of ee_chi, mu, g_loc, and zeta
    
    ee_chi = second_invariant(U,dx) # second invariant of the strain rate in 
    # transformed coordinate system
    
    mu = (ee_chi/L)/(gg+config.dgg) # effective coefficient of friction
    
    mu = np.abs(mu)
    if np.min(mu)<0:
        sys.exit('mu less than 0!')
    
    # Calculate g_loc using a regularization that approximates the next two lines.
    # g_loc = config.secsYear*np.sqrt(pressure(H)/config.rho)*(mu-config.muS)/(mu*config.b*config.d)
    # g_loc[g_loc<0] = 0
    k = 50 # smoothing factor (small number equals more smoothing?)
   
    f = [quad(calc_df,1e-4,x,args=(config.muS,k))[0] for x in mu] 
    f = np.abs(f)
    if np.min(f)<0:
        sys.exit('g_loc less than 0!')
    
    g_loc = config.secsYear*np.sqrt(pressure(H)/config.rho)*f/(config.b*config.d)
 

    # Regularization of abs(mu-muS)
    f_mu = 2/k*np.log(1+np.exp(k*(mu-config.muS)))-mu+config.muS-2/k*np.log(1+np.exp(-k*config.muS))

    zeta = f_mu/(config.A**2*config.d**2)
    # Essentially Equation 19 in Amundson and Burton (2018)
    
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

    return(res)


#%%
def calc_muW(muW, H, W, U):
    '''
    Compares the velocity from the longitudinal flow model to the width-averaged
    velocity from the transverse profile, for given coefficients of friction
    and geometry. For low velocities, there is relatively little deformation and
    muW will be small. For high velocities, muW will be high. There is no slip
    along the fjord walls since the model doesn't assume an upper end to muW.

    Currently assumes no slip.

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
def calc_H(H, data, H_prev):    
    '''
    Compute the residual for the thickness differential equation.

    Parameters
    ----------
    H : initial guess for the thickness
    data : current state of the model object
    H_prev : thickness profile from the previous time step
    
    Returns
    -------
    res : residual for the thickness differential equation
    '''
    
    # extract variables from the model object
    x = data.x
    x_ = data.x_
    dx = data.dx
    dt = data.dt
    W = data.W
    L = data.L
    U = data.U
    B = data.B
    
    # defined for simplicity later
    beta = U[0]+x_*(U[-1]-U[0])
          
    a_left = dt/(2*dx*L)*(beta[1:] - W[:-1]/W[1:]*(U[1:-1]+U[:-2]))
    a_left[-1] = dt/(dx*L)*(beta[-1]-0.5*W[-2]/W[-1]*(U[-2]+U[-3]))
    
    a = 1 + dt/(2*dx*L)*(U[2:]+U[1:-1])
    a[-1] = 1+dt/(dx*L)*(-beta[-1]+0.5*(U[-1]+U[-2])) # check sign in front of beta!
    a = np.append(1+beta[0]*dt/(L*dx)-dt/(2*L*dx)*(U[1]+U[0]), a)
    
    a[0] = 1 + dt/(dx*L)*(beta[0] - (U[0]+U[1])/2) 
    
    a_right = -dt/(2*dx*L)*beta[1:]
    a_right[0] = dt/(L*dx)*(-beta[0]+(U[2]+U[1])/2*W[1]/W[0])
    
    T = B*dt + H_prev
    
    
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray()
    
    # set d^2{H}/dx^2=0 across the first three grid points (to deal with upwind scheme)
    D[0,0] = 1
    D[0,1] = -2
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
    muW : effective coefficient of friction along the fjord walls
    H : ice melange thickness [m]
    
    Returns
    -------
    y : transverse coordinate [m]
    u : velocity at y, assuming no slip along the boundary [m s^-1]
    u_mean : mean velocity across the profile, assuming no slip along the boundary [m s^-1]

    '''
    
    n_pts = 101 # number of points in half-width
    y = np.linspace(0,W/2,n_pts) # location of points
    
    dy = y[1] # grid spacing

    mu = muW*(1-2*y/W) # variation in mu across the fjord, assuming quasi-static flow

    y_c = W/2*(1-config.muS/muW) # critical value of y for which mu is no longer greater 
    # than muS; although flow occurs below this critical value, it is needed for 
    # computing g_loc (below)
       
    zeta = np.sqrt(np.abs(mu-config.muS))/(config.A*config.d)
    
    g_loc = np.zeros(len(y))
    g_loc[y<y_c] = config.secsYear*np.sqrt(pressure(H)/config.rho)*(mu[y<y_c]-config.muS)/(mu[y<y_c]*config.b*config.d) # local granular fluidity
    
    
    # First solve for the granular fluidity. we set dg/dy = 0 at
    # y = 0 and at y = W/2. Because mu is known, this does not need to be done
    # iteratively (as it was done in the along-flow direction.)    
    
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
def second_invariant(U,dx):
    '''
    Calculate the second invariant of the strain rate with respect to chi (dimensionless grid)

    Parameters
    ----------
    U : velocity profile [m/yr]
    dx : grid spacing

    Returns
    -------
    ee_chi : second invariant of the strain rate tensor [1/yr]
    '''
    
    ee_chi = np.sqrt((np.diff(U)/dx)**2/2)

    return(ee_chi)


#%% calculate the effective pressure driving flow
def pressure(H):
    '''
    Calculate the difference between the depth-averaged glaciostatic and 
    hydrostatic pressures, assuming at flotation.    

    Parameters
    ----------
    H : thickness profile [m]
    
    Returns
    -------
    P : pressure [Pa]
    '''
    
    P = 0.5*config.rho*config.g*(1-config.rho/config.rho_w)*H
    
    return(P)


#%%
def basic_figure(n,dt):
    '''
    Sets up the basic figure for plotting. 

    Returns
    -------
    axes handles ax1, ax2, ax3, ax4, ax5, and ax_cbar for the 5 axes and colorbar
    '''
    
    color_id = np.linspace(0,1,n)

    fig_width = 12
    fig_height = 6.5
    plt.figure(figsize=(fig_width,fig_height))
    
    ax_width = 3/fig_width
    ax_height = 2/fig_height
    left = 1/fig_width
    bot = 0.5/fig_height
    ygap = 0.75/fig_height
    xgap= 1/fig_width
    
    
    xmax = 15000
    vmax = 50
    
    ax1 = plt.axes([left, bot+ax_height+2.25*ygap, ax_width, ax_height])
    ax1.set_xlabel('Longitudinal coordinate [m]')
    ax1.set_ylabel('Speed [m/d]')
    ax1.set_ylim([0,50])
    ax1.set_xlim([0,xmax])
    
    ax2 = plt.axes([left+ax_width+xgap, bot+ax_height+2.25*ygap, ax_width, ax_height])
    ax2.set_xlabel('Longitudinal coordinate [m]')
    ax2.set_ylabel('Elevation [m]')
    ax2.set_ylim([-75, 100])
    ax2.set_xlim([0,xmax])
    
    ax3 = plt.axes([left, bot+1.25*ygap, ax_width, ax_height])
    ax3.set_xlabel('Longitudinal coordinate [m]')
    ax3.set_ylabel('$g^\prime$ [a$^{-1}]$')
    ax3.set_ylim([0, 20])
    ax3.set_xlim([0,xmax])
    
    ax4 = plt.axes([left+ax_width+xgap, bot+1.25*ygap, ax_width, ax_height])
    ax4.set_xlabel('Longitudinal coordinate [m]')
    ax4.set_ylabel('$\mu_w$')
    ax4.set_ylim([0, 3])
    ax4.set_xlim([0,xmax])
    
    ax5 = plt.axes([left+2*(ax_width+xgap), bot+1.25*ygap, 0.75*ax_width, 2*ax_height+ygap])
    ax5.set_xlabel('Transverse coordinate [m]')
    ax5.set_ylabel(r'Speed at $\chi=0.5$ [m/d]')
    ax5.set_xlim([0,4000])
    ax5.set_ylim([0,vmax])
    
    ax_cbar = plt.axes([left, bot, 2*(ax_width+xgap)+0.75*ax_width, ax_height/15])
    
    cbar_ticks = np.linspace(0, (n-1)*dt, 11, endpoint=True)
    cmap = matplotlib.cm.viridis
    bounds = cbar_ticks
    norm = matplotlib.colors.Normalize(vmin=0, vmax=(n-1)*dt)
    cb = matplotlib.colorbar.ColorbarBase(ax_cbar, cmap=cmap, norm=norm,
                                    orientation='horizontal')#,extend='min')
    cb.set_label("Time [a]")

    axes = (ax1, ax2, ax3, ax4, ax5, ax_cbar)
    
    return(axes, color_id)

#%%
def plot_basic_figure(data, axes, color_id, k):
    '''
    Take the current state of the model object and plot the basic figure.
    '''
    
    # extract variables from model object
    X = data.X
    X_ = data.X_
    U = data.U
    H = data.H
    W = data.W
    gg = data.gg
    muW = data.muW
    W = data.W

    # compute transverse velocity profile
    ind = int(len(W)/2) # index of midpoint in fjord
    y, u_transverse, _ = transverse(W[ind],muW[ind],H[ind])    

    ax1, ax2, ax3, ax4, ax5, ax_cbar = axes
    ax1.plot(X,U/config.daysYear,color=plt.cm.viridis(color_id[k]))
    ax2.plot(np.append(X_,X_[::-1]),np.append(-config.rho/config.rho_w*H,(1-config.rho/config.rho_w)*H[::-1]),color=plt.cm.viridis(color_id[k]))
    ax3.plot(X_,gg,color=plt.cm.viridis(color_id[k]))
    ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[k]))
    ax5.plot(np.append(y,y+y[-1]),np.append(u_transverse,u_transverse[-1::-1])/config.daysYear,color=plt.cm.viridis(color_id[k]))


