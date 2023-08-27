# write code to refine grid

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.integrate import trapz

import config

from matplotlib import pyplot as plt
import matplotlib

import sys

import pickle

import os

#%%
class glaciome:
    '''
    The model class contains model variables, which simplifies passing variables
    into and out of functions, and methods for model spinup, diagnostic solutions,
    and implicit time steps.
    '''
    
    def __init__(self, n_pts, dt, L, Ut, Uc, Ht, X_fjord, W_fjord):
        '''
        Initialize an object at t=0.

        Parameters
        ----------
        n_pts : number of grid points
        L : ice melange length [m]
        Ut : terminus velocity [m/a]
        Uc : calving rate [m/a]
        Ht = terminus thickness
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
        self.H = config.d*(1-self.x_) + config.d
        self.update_H_endpoints()
        
        self.Ht = Ht
        self.Ut = Ut
        self.Uc = Uc
        self.U0 = self.Ut + self.Uc*(self.Ht/self.H0-1)
        self.U = self.U0*np.exp(-self.x/0.5)
        self.gg = 1*(1-self.x_)
        self.muW = config.muW_*np.ones(len(self.H)-1)
        
        # set initial values for width and mass balance
        self.B = -0.1*config.daysYear # initialize mass balance rate as -1 m/d
        
        # time step and initial time
        self.dt = dt
        self.t = 0 # current model time
        
        self.subgrain_deformation = 'n'
        self.width_interpolator = create_width_interpolator(X_fjord, W_fjord)
        self.W = np.array([self.width_interpolator(x) for x in self.X_])
        self.update_width()
        self.tauX = 0 # drag force N/m; later divide by L to cast in terms of a stress; note units have 1/s^2, but it will be divided by gravity so it's okay
        
        self.transient = 1 # 1 for transient simulation, 0 to try to solve for steady-state
        
        

    def diagnostic(self):
        '''
        Run this if a spinup.pickle file does not already exist. Requires a 
        glaciome object to have already been created.
        
        Spinup ultimately involves just solving the diagnostic equations for a
        given geometry. In order to determine a good starting point for solving
        the diagnostic equations, the code first alternates between solving for 
        the granular fluidity and solving for the velocity. After some 
        convergence, it then solves for both of them simultaneously.
        '''
        
    
        # alternate solving for granular fluidity and velocity in order to get a good starting point
        print('Iterating between granular fluidity and velocity in order to find a good starting point')
        residual = self.Ut*np.ones(self.x.shape) # just some large values for the initial residuals
        
        while np.max(np.abs(residual))>1:
            print('Velocity residual: ' + "{:.2f}".format(np.max(np.abs(residual))) + ' m/a')
            
            # solve for granular fluidity
            result = root(calc_gg, self.gg, (self), method='hybr')
            self.gg = result.x
            
            # solve for velocity and compute residual
            result = root(calc_U, self.U, (self), method='hybr')
            U_new = result.x
            residual = self.U-U_new
            self.U = U_new
        
        print('Done with initial iterations. Solving diagnostic equations.')
        
        # now simultaneous solve for velocity and granular fluidity
        Ugg = np.concatenate((self.U,self.gg))
        result = root(diagnostic, Ugg, (self), method='hybr')
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
    
    def prognostic(self):
        '''
        Use an implicit time step to determine the velocity, granular fluidity, 
        thickness, and length.
        '''
        
        # The previous thickness and length are required.
        H_prev = self.H
        L_prev = self.L
        
        UggHL = np.concatenate((self.U,self.gg,self.H,[self.L])) # starting point for solving differential equations
        
        result = root(solve_prognostic, UggHL, (self, H_prev, L_prev), method='hybr')#, options={'maxiter':int(1e6)}) #Since we are using an implicit time step to update $H$, we must solve $2N+1$ equations: $N+1$ equations come from the stress balance equation (Equation \ref{eq:stress_balance_stretched}) and associated boundary conditions, and $N$ equations come from the mass continuity equation (Equation \ref{eq:mass_continuity_stretched}).
        print('result status: ' + str(result.status))
        print('result success: ' + str(result.success))
        print('result message: ' + str(result.message))
        
        
        self.U = result.x[:len(self.x)]
        self.gg = result.x[len(self.x):2*len(self.x)-1]
        self.H = result.x[2*len(self.x)-1:-1]
        self.L = result.x[-1]
        
        # Update the dimensional grid and the width
        self.X = self.x*self.L + self.X[0] + (self.Ut-self.Uc)*self.dt
        self.X_ = (self.X[:-1]+self.X[1:])/2
        self.W = np.array([self.width_interpolator(x) for x in self.X_])

        # Update the coefficient of friction along the fjord walls.
        self.update_muW()
        
        # Update the time stored within the model object.
        self.t += self.dt
        
        
    def update_muW(self):
        '''
        The coefficient of friction along the fjord walls is calculated when
        iteratively solving for the velocity, but at that point is not directly
        saved in the model object. Here it is calculated and stored in the model 
        object by using the current values of the velocity, thickness, and width.
        '''
        
        H_ = (self.H[:-1]+self.H[1:])/2
        if self.subgrain_deformation=='n':
            H_ = deformational_thickness(H_)
            
        W_ = (self.W[:-1]+self.W[1:])/2
        for k in range(len(self.H)-1):
            self.muW[k] = fsolve(calc_muW, config.muW_, (H_[k],W_[k],self.U[k+1])) # excluding first and last grid points, where we are prescribing boundary conditions
            self.muW[k] = np.min((self.muW[k], config.muW_max))
            
    def update_width(self):  
        '''
        Update the melange width by interpolating the fjord walls at the grid
        points, X. Currently only set up to handle constant width. 
        '''       
        
        self.W = self.width_interpolator(self.X_)
        self.W0 = self.width_interpolator(self.X[0])
        self.WL = self.width_interpolator(self.X[-1])
        
    def update_H_endpoints(self):
        self.H0 = 1.5*self.H[0]-0.5*self.H[1]
        self.HL = 1.5*self.H[-1]-0.5*self.H[-2]
        
    
            
    def save(self,file_name):
        '''
        Save the model output at time step k.
        '''
        if not os.path.exists('output'): 
            os.mkdir('output')
        
        #file_name = 'output/output_' + str(k).zfill(4) + '.pickle'
        with open(file_name, 'wb') as file:
            pickle.dump(self, file)
            file.close()
    
    
    
#%%          
def solve_prognostic(UggHL, data, H_prev, L_prev):
    '''
    Computes the residual for the velocity, granular fluidity, thickness, and
    length differential equations. The residual is minimized in model.implicit.
    '''
    
    # extract current value for U, gg, H, L
    data.U = UggHL[:len(data.x)]
    data.gg = UggHL[len(data.x):2*len(data.x)-1]
    data.H = UggHL[2*len(data.x)-1:-1]
    data.H0 = 1.5*data.H[0]-0.5*data.H[1]
    data.HL = 1.5*data.H[-1]-0.5*data.H[-2]
    data.U0 = data.Ut + data.Uc*(data.Ht/data.H0-1)
    data.L = UggHL[-1]

    data.dLdt = (data.L-L_prev)/data.dt

    # introduce some scales for non-dimensionalizing the differential equations
    Hscale = 100 # [m]
    Lscale = 1e4 # [m]
    Uscale = 0.5e4 # [m/a]
    Tscale = Lscale/Uscale # [a]
    
    
    resU = calc_U(data.U, data) * (Lscale/(Uscale*Hscale**2*Tscale))
    resgg = calc_gg(data.gg, data) * Tscale**2
    resH = calc_H(data.H, data, H_prev) * Tscale / Hscale # scaling here might be wrong, check how equation is formulated!

    resHc = (1.5*data.H[-1]-0.5*data.H[-2]-config.Hc) / Hscale
    
    # append residuals into single variable to be minimized
    resUggHL = np.concatenate((resU, resgg, resH, [resHc]))    
    
    
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
    muW = data.muW
    U0 = data.U0
        
    # determine H and W on the grid in order to calculate the coefficient of friction along the fjord walls
    # note that these start at the second grid point
    H_ = (H[:-1]+H[1:])/2 # thickness on the grid, excluding the first and last grid points
    W_ = (W[:-1]+W[1:])/2 # width on the grid, excluding the first and last grid points        
    
    
    # create new variable, nu, to simplify discretization
    if data.subgrain_deformation=='n':
        Hd = deformational_thickness(H)
        Hd_ = deformational_thickness(H_)
        nu = H*Hd/gg
        #nu = H*Hd/(gg+config.dgg)
        #nu = H*Hd/np.sqrt(gg**2+config.dgg**2)
        
    else:
        nu = H**2/(gg+config.dgg) 
    
    
    ## THIS FOR LOOP CAN BE PARALELLIZED LATER
    # calculate muW given the current velocity profile
    for k in range(len(H_)):
        if data.subgrain_deformation=='n':
            muW[k] = fsolve(calc_muW, config.muW_, (Hd_[k],W_[k],U[k+1])) # excluding first and last grid points, where we are prescribing boundary conditions
        else: 
            muW[k] = fsolve(calc_muW, config.muW_, (H_[k],W_[k],U[k+1])) # excluding first and last grid points, where we are prescribing boundary conditions
        
        muW[k] = np.min((muW[k],config.muW_max))
    
    a_left = nu[:-1]/(dx*L)**2    
    a_left = np.append(a_left,-1/(dx*L))
    
    a = np.ones(len(U))/(dx*L)
    a[1:-1] = -(nu[1:] + nu[:-1])/(dx*L)**2
    
    a_right = nu[1:]/(dx*L)**2
    a_right = np.append(0,a_right)
    
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray() 
    
    
    if data.subgrain_deformation=='n':
        T = H_*(H[1:]-H[:-1])/(dx*L) + 2*H_*Hd_/W_*muW*np.sign(U[1:-1])
    else:
        T = H_*(H[1:]-H[:-1])/(dx*L) + 2*H_**2/W_*muW*np.sign(U[1:-1])
     
    # upstream boundary condition
    T = np.append(U0/(dx*L),T)
    
    # downstream boundary condition; longitudinal resistive stress equals
    # difference between glaciostatic and hydrostatic stresses
    if data.subgrain_deformation=='n':
        T = np.append(T,0)
    else:
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
    df_ = df*logistic(mu,muS,k) # modified df/d(mu)
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
    
    if data.subgrain_deformation=='n':
        H = deformational_thickness(H)
    
    
    # calculate current values of ee_chi, mu, g_loc, and zeta

    #ee_chi = second_invariant(U,dx) # second invariant of the strain rate in 
    # transformed coordinate system
    ee_chi = second_invariant(U,dx)
    
    
    
    mu = (ee_chi/L+config.deps)/gg # option 1
    #mu = np.sqrt((ee_chi/L)**2+config.deps**2)/gg # option 2
    # mu = (ee_chi/L)/(1-np.exp(-(ee_chi/L)/config.deps)) / gg # option 3
    
    
    mu = np.abs(mu)
    if np.min(mu)<0:
        sys.exit('mu less than 0!')
    
    # Calculate g_loc using a regularization that approximates the next two lines.
    # g_loc = config.secsYear*np.sqrt(pressure(H)/config.rho)*(mu-config.muS)/(mu*config.b*config.d)
    # g_loc[g_loc<0] = 0
    
    #k = 50 # smoothing factor (small number equals more smoothing?)
   
    #f = [quad(calc_df,1e-4,x,args=(config.muS,k))[0] for x in mu] 
    #f = np.abs(f)
    #if np.min(f)<0:
    #    sys.exit('g_loc less than 0!')

    f = 1-config.muS/mu
    g_loc = config.secsYear*np.sqrt(pressure(H)/config.rho)*f/(config.b*config.d)
    #print(g_loc)
    g_loc[g_loc<0] = 0
    
    k = 50
    # Regularization of abs(mu-muS)
    f_mu = 2/k*np.logaddexp(0,k*(mu-config.muS)) - 2/k*np.logaddexp(0,-k*config.muS) + config.muS - mu
    #f_mu = np.abs(mu-config.muS)

    zeta = f_mu/(config.A**2*config.d**2)
    # Essentially Equation 19 in Amundson and Burton (2018)
    
    # construct equation Dx=T
    # boundary conditions:
    #    dg/dx=0 is the soft boundary condition recommended by Henann and Kamrin (2013)
    
    bc = 'first-order' # specify whether boundary condition should be 'first-order' accurate or 'second-order' accurate
    
    
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
    muW will be small. For high velocities, muW will be high. 

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
    
    _, _, u_mean = transverse(W, muW, H)
    
    # take into account that flow might be in the negative direction
    if U<0:
        u_mean = -u_mean
        
    du = np.abs(U-u_mean)    
    
    return(du)

#%%    
def calc_H(H, data, H_prev):#, Qf):    
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
    x_ = data.x_
    dx = data.dx
    dt = data.dt
    W = data.W
    L = data.L
    U = data.U
    B = data.B
    Ut = data.Ut
    Uc = data.Uc
    dLdt = data.dLdt
    
    # defined for simplicity later  
    #beta = U[0]+x_*dLdt
    beta = data.transient*(Ut - Uc + x_*dLdt)
    
    a_left = 1/(2*dx*L)*(beta[1:] - W[:-1]/W[1:]*(U[1:-1]+U[:-2]))
    a_left[-1] = 1/(dx*L)*(beta[-1]-0.5*W[-2]/W[-1]*(U[-2]+U[-3]))
    
    a = data.transient*1/dt + 1/(2*dx*L)*(U[2:]+U[1:-1])
    a[-1] = data.transient*1/dt+1/(dx*L)*(-beta[-1]+0.5*(U[-1]+U[-2])) 
    a = np.append(data.transient*1/dt+beta[0]/(L*dx)-1/(2*L*dx)*(U[1]+U[0]), a)
    
    a[0] = data.transient*1/dt + 1/(dx*L)*(beta[0] - (U[0]+U[1])/2) 
    
    a_right = -1/(2*dx*L)*beta[1:]
    a_right[0] = 1/(L*dx)*(-beta[0]+(U[2]+U[1])/2*W[1]/W[0])
    
    T = B + data.transient*H_prev/dt
    
    
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
def transverse(W, muW, H): 
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
    
    
    # Compute residual for the granular fluidity. we set dg/dy = 0 at
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
def calc_steady_state(HL_values, data):
    '''
    Calculates steady-state profiles (dL/dt=0, dH/dt=0, Ut=Uc, dU/dx=0)

    Parameters
    ----------
    HL : Array that contains the initial guesses for H and L [m]
    data : glaciome object

    Returns
    -------
    res : Residual of the differential equation; this needs to be minimized to
    determine the profiles

    Note that there are the following number of unknowns, which requires the
    same number of equations.
    
    H: N-1 unknowns
    L: 1 unknown
    total: N unknowns
    '''
    
    data.H = HL_values[0:-1]
    data.L = HL_values[-1]
    
    # thicknesses at x = 0 and x = 1
    data.update_H_endpoints()
    data.update_width()
    
    # because steady-state, Ut = Uc
    data.U0 = data.Ut*data.Ht/data.H0
    
    # subtract the grain scale
    Hd = deformational_thickness(data.H) # between the grid points
    Hd_ = (Hd[1:]+Hd[:-1])/2 # on the grid points
    
    data.X_ = data.x_*data.L # coordinates between the grid points
   
    data.W = data.width_interpolator(data.X_)
    W_ = (data.W[1:]+data.W[:-1])/2 # on the grid points, excluding end points
    
    H_ = (data.H[1:]+data.H[:-1])/2 # on the grid points, excluding end points
    
    dx = data.dx
    B = data.B
    
    for k in range(len(Hd_)):
        data.muW[k] = fsolve(calc_muW, config.muW_, (Hd_[k],W_[k],data.U0)) # excluding first and last grid points, where we are prescribing boundary conditions
    
        
    # scales
    Hscale = 100
    Uscale = 5000
    
    # N equations
    resH = ((data.H[1:]-data.H[:-1])/(data.L*dx) + Hd_/W_*data.muW - 2*data.tauX/(data.L*pressure(H_))) # N-2 equations 
    resHend = (data.H[-1]-config.d)/Hscale # 1 equation
    
    BW_int = trapz(B*np.concatenate(([data.W0],data.W,[data.WL])),np.concatenate(([0],data.X_,[data.L]))) # integral of B*W*dx
    resB = (BW_int+data.U0*(data.H0*data.W0-data.HL*data.WL))/Uscale # 1 equation
    
    
    res = np.concatenate((resH,[resHend],[resB]))
    
    return(res)


#%%
def create_width_interpolator(X_fjord, W_fjord):
    '''
    The width on the staggered grid must be determined at each time step. This 
    function creates an interpolator based on a given fjord geometry.

    Parameters
    ----------
    X_fjord : Longitudinal coordinate [m]
    W_fjord : Fjord width [m]

    Returns
    -------
    width_interpolator
    '''
    
    width_interpolator = interp1d(X_fjord, W_fjord, fill_value='extrapolate') 

    return(width_interpolator)




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

#%%
def deformational_thickness(H):
    '''
    Subtract the grain size so that deformation only occurs above this value. 
    In order to prevent negative pressures, this is done by using an integrated
    logistic function

    Parameters
    ----------
    H : ice melange thickness [m]
    
    Returns
    -------
    Hd : ice melange thickness minus the grain size [m]

    '''
    
    k = 0.25
    Hd = np.logaddexp(0,k*(H-config.d))/k
    #Hd = np.array([np.max([0,x-config.d]) for x in H])
    
    return(Hd)
    

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
    vmax = 120
    
    ax1 = plt.axes([left, bot+ax_height+2.25*ygap, ax_width, ax_height])
    ax1.set_xlabel('Longitudinal coordinate [m]')
    ax1.set_ylabel('Speed [m/d]')
    ax1.set_ylim([0,vmax])
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
    ax5.set_xlim([-4000,4000])
    ax5.set_ylim([0,vmax])
    
    ax_cbar = plt.axes([left, bot, 2*(ax_width+xgap)+0.75*ax_width, ax_height/15])
    
    #cbar_ticks = np.linspace(0, (n-1)*dt, 11, endpoint=True)
    cmap = matplotlib.cm.viridis
    #bounds = cbar_ticks
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
    X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
    U = data.U
    H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))
    W = data.W
    gg = np.concatenate(([1.5*data.gg[0]-0.5*data.gg[1]],data.gg,[1.5*data.gg[-1]-0.5*data.gg[-2]]))
    muW = np.concatenate(([1.5*data.muW[0]-0.5*data.muW[1]],data.muW,[1.5*data.muW[-1]-0.5*data.muW[-2]]))
    muW[0] = np.min((muW[0], config.muW_max))
    muW[-1] = np.min((muW[-1], config.muW_max))
    W = data.W
    
    X = X-X[0]
    X_ = X_-X_[0]
    # compute transverse velocity profile
    # need to clean this up later
    ind = int(len(W)/2) # index of midpoint in fjord
    
    if data.subgrain_deformation=='n':
        #y, u_transverse, _ = transverse(W[ind],muW[ind],deformational_thickness(H[ind]))
        y, u_transverse, _ = transverse(W[ind],muW[ind],np.max([0,H[ind]-config.d]))

    else:
        y, u_transverse, _ = transverse(W[ind],muW[ind],H[ind])
        
    u_slip = U[ind]-np.mean(u_transverse)
    u_transverse += u_slip

    ax1, ax2, ax3, ax4, ax5, ax_cbar = axes
    ax1.plot(X,U/config.daysYear,color=plt.cm.viridis(color_id[k]))
    ax2.plot(np.append(X_,X_[::-1]),np.append(-config.rho/config.rho_w*H,(1-config.rho/config.rho_w)*H[::-1]),color=plt.cm.viridis(color_id[k]))
    ax3.plot(X_,gg,color=plt.cm.viridis(color_id[k]))
    ax4.plot(X,muW,color=plt.cm.viridis(color_id[k]))
    ax5.plot(np.append(y-y[-1],y),np.append(u_transverse,u_transverse[-1::-1])/config.daysYear,color=plt.cm.viridis(color_id[k]))


    
