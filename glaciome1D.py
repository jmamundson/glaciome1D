# write code to refine grid

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags
from scipy.optimize import root
from scipy.interpolate import interp1d
from scipy.integrate import trapz

from matplotlib import pyplot as plt
import matplotlib

import sys

import pickle

import os

#%%
class constants:
    
    def __init__(self):
        self.rho = 917.       # density of ice
        self.rho_w = 1028.    # density of water
        self.g = 9.81         # gravitational acceleration

        self.secsDay = 86400.
        self.daysYear = 365.25
        self.secsYear = self.secsDay*self.daysYear

#%%
class parameters:
    '''
    Parameters for the nonlocal granular fluidity rheology
    '''
    
    def __init__(self):
        self.deps = 0.1     # finite strain rate parameter [a^-1]; might work best to start with the large during spin up, then decrease it?
        
        self.d = 25 # characteristic iceberg size [m]
        self.A = 0.5 
        self.b = 2e4
        self.muS = 0.2
        self.muW_ = 0.5 # guess for muW iterations
        self.muW_max = 1 # maximum value for muW 
        

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
        W : ice melange width [m]
        '''
        
        self.constants = constants()
        self.param = parameters()
        
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
        
        self.width_interpolator = create_width_interpolator(X_fjord, W_fjord)
        self.W = np.array([self.width_interpolator(x) for x in self.X_])
        self.W0 = self.width_interpolator(self.X[0])
        self.WL = self.width_interpolator(self.X[-1])

        self.H = fsolve(quasistatic_thickness, 2*self.param.d*np.ones(len(self.x_)), (self))#, method='hybr')
        #self.H = self.param.d*(1-self.x_) + self.param.d
        self.H0 = 1.5*self.H[0]-0.5*self.H[1]
        self.HL = self.param.d
        
        self.Ht = Ht
        self.Ut = Ut
        self.Uc = Uc
        self.U0 = self.Ut + self.Uc*(self.Ht/self.H0-1)
        self.U = self.U0*np.ones(len(self.x))#np.exp(-self.x/0.5)
        #self.gg = 1*(1-self.x_)
        
        self.muW = self.param.muW_*np.ones(len(self.U)-2)
        self.gg = self.param.deps/self.muW[0]*np.ones(len(self.H))
        
        # set initial values for width and mass balance
        self.B = -0.5*self.constants.daysYear # initialize mass balance rate as -1 m/d
        
        # time step and initial time
        self.dt = dt
        self.t = 0 # current model time
        
        self.subgrain_deformation = 'n'

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
        
        # simultaneously solve for velocity and granular fluidity
        Ugg = np.concatenate((self.U,self.gg))
        result = root(solve_diagnostic, Ugg, (self), method='hybr')
        self.U = result.x[:len(self.x)]
        self.gg = result.x[len(self.x):]
        
    
    def prognostic(self):
        '''
        Use an implicit time step to determine the velocity, granular fluidity, 
        thickness, and length.
        '''
        
        # The previous thickness and length are required.
        H_prev = self.H
        L_prev = self.L
        
        self.X[0] += (self.Ut-self.Uc)*self.dt # use explicit time step to find new position X0.
        
        UggHL = np.concatenate((self.U,self.gg,self.H,[self.L])) # starting point for solving differential equations
        
        result = root(solve_prognostic, UggHL, (self, H_prev, L_prev), method='hybr')#, options={'maxiter':int(1e6)}) #Since we are using an implicit time step to update $H$, we must solve $2N+1$ equations: $N+1$ equations come from the stress balance equation (Equation \ref{eq:stress_balance_stretched}) and associated boundary conditions, and $N$ equations come from the mass continuity equation (Equation \ref{eq:mass_continuity_stretched}).
        #print('result status: ' + str(result.status))
        #print('result success: ' + str(result.success))
        #print('result message: ' + str(result.message))
        
        self.U = result.x[:len(self.x)]
        self.gg = result.x[len(self.x):2*len(self.x)-1]
        self.H = result.x[2*len(self.x)-1:-1]
        self.L = result.x[-1]
        
        # Update the time stored within the model object.
        self.t += self.dt
        
    def refine_grid(self, n_pts):
        
        x = np.linspace(0,1,n_pts)
        x_ = (x[:-1]+x[1:])/2
        
        X = x*self.L + self.X[0]
        X_ = x_*self.L + self.X[0]
        
        U_interpolator = interp1d(self.X, self.U, kind='linear', axis=-1, fill_value='extrapolate')
        H_interpolator = interp1d(self.X_, self.H, kind='linear', axis=-1, fill_value='extrapolate')
        W_interpolator = interp1d(self.X_, self.W, kind='linear', axis=-1, fill_value='extrapolate')
        gg_interpolator = interp1d(self.X_, self.gg, kind='linear', axis=-1, fill_value='extrapolate')
        muW_interpolator = interp1d(self.X[1:-1], self.muW, kind='linear', axis=-1, fill_value='extrapolate')
        
        self.U = U_interpolator(X)
        self.H = H_interpolator(X_)
        self.W = W_interpolator(X_)
        self.gg = gg_interpolator(X_)
        self.muW = muW_interpolator(X[1:-1])
        
        self.dx = x[1]
        self.x = x
        self.x_ = x_
        self.X = X
        self.X_ = X_
            
    
    def steadystate(self):
        print('Solving diagnostic equation.')
        self.diagnostic() # solve diagnostic equation first to ensure that model is consistent
        L_old = self.L
        dL = 1000 # just initiating the change in length with some large value
        
        k = 1 # initiate counter
        t = 0
        
        print('Solving prognostic equations.')
        while np.abs(dL)>50: # may need to adjust if final solution looks noisy      
            self.dt = 0.25*self.dx*self.L/np.max(self.U) # use an adaptive time step, loosely based on CFL condition (for explicit schemes)
            self.prognostic()
            t += self.dt
            
            if (k%10) == 0:
                X_ = np.concatenate(([self.X[0]],self.X_,[self.X[-1]]))
                H = np.concatenate(([self.H0],self.H,[1.5*self.H[-1]-0.5*self.H[-2]]))
                
                print('Step: ' + str(int(k)) + ' years')   
                print('Length: ' + "{:.2f}".format(self.L) + ' m')
                print('Change in length: ' + "{:.2f}".format(self.L-L_old) + ' m') # over 10 time steps
                print('Volume: ' + "{:.4f}".format(trapz(H, X_)*4000/1e9) + ' km^3')
                print('H_L: ' + "{:.2f}".format(1.5*self.H[-1]-0.5*self.H[-2]) + ' m') 
                print('CFL: ' + "{:.4f}".format(np.max(self.U)*self.dt/(self.dx*self.L)))
                print(' ')
                dL = self.L-L_old
                L_old = self.L
                
            k += 1
        
        self.transient = 0
        print('Steady-state solve.')
        self.prognostic()
        self.transient = 1    
        
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

    # Update the dimensional grid and the width
    data.X = data.x*data.L + data.X[0]
    data.X_ = (data.X[:-1]+data.X[1:])/2
    data.W = data.width_interpolator(data.X_)

    # update endpoints, since W and H are on the staggered grid
    data.W0 = data.width_interpolator(data.X[0])
    data.WL = data.width_interpolator(data.X[-1])
    data.H0 = 1.5*data.H[0]-0.5*data.H[1]
    data.HL = 1.5*data.H[-1]-0.5*data.H[-2]

    # introduce some scales for non-dimensionalizing the differential equations
    Hscale = 100 # [m]
    Lscale = 1e4 # [m]
    Uscale = 0.5e4 # [m/a]
    Tscale = Lscale/Uscale # [a]
    
    resU = calc_U(data.U, data) * (Lscale/(Uscale*Hscale**2*Tscale))
    resgg = calc_gg(data.gg, data) * Tscale**2
    resH = calc_H(data.H, data, H_prev) * Tscale / Hscale # scaling here might be wrong, check how equation is formulated!

    resHc = (1.5*data.H[-1]-0.5*data.H[-2]-data.param.d) / Hscale
    
    # append residuals into single variable to be minimized
    resUggHL = np.concatenate((resU, resgg, resH, [resHc]))    
    
    
    return(resUggHL)




#%%
def solve_diagnostic(Ugg,data):
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
        Hd = deformational_thickness(H,data)
        Hd_ = deformational_thickness(H_,data)
        nu = H*Hd/gg     
    else:
        nu = H**2/gg 
    
    
    ## THIS FOR LOOP CAN BE PARALELLIZED LATER
    # calculate muW given the current velocity profile
    for k in range(len(H_)):
        if data.subgrain_deformation=='n':
            muW[k] = fsolve(calc_muW, data.param.muW_, (Hd_[k],W_[k],U[k+1],data)) # excluding first and last grid points, where we are prescribing boundary conditions
        else: 
            muW[k] = fsolve(calc_muW, data.param.muW_, (H_[k],W_[k],U[k+1],data)) # excluding first and last grid points, where we are prescribing boundary conditions
        
        muW[k] = np.min((muW[k],data.param.muW_max))
    
    data.muW = muW
    
    a_left = nu[:-1]/(dx*L)**2    
    a_left = np.append(a_left,-1/(dx*L))
    
    a = np.ones(len(U))/(dx*L)
    a[1:-1] = -(nu[1:] + nu[:-1])/(dx*L)**2
    
    a_right = nu[1:]/(dx*L)**2
    a_right = np.append(0,a_right)
    
    diagonals = [a_left,a,a_right]
    D = diags(diagonals,[-1,0,1]).toarray() 
    
    
    if data.subgrain_deformation=='n':
        T = H_*(H[1:]-H[:-1])/(dx*L) + 2*H_*Hd_/W_*muW*np.sign(U[1:-1]) - (data.tauX/data.L)/(data.constants.rho*data.constants.g*(1-data.constants.rho/data.constants.rho_w))
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
        H = deformational_thickness(H,data)
    
    
    # calculate current values of ee_chi, mu, g_loc, and zeta

    #ee_chi = second_invariant(U,dx) # second invariant of the strain rate in 
    # transformed coordinate system
    ee_chi = second_invariant(U,dx)
    
    mu = (ee_chi/L+data.param.deps)/gg # option 1
    #mu = np.sqrt((ee_chi/L)**2+data.param.deps**2)/gg # option 2 !!! seems best?
    #mu = (ee_chi/L)/(1-np.exp(-(ee_chi/L)/data.param.deps)) / gg # option 3
    
    
    mu = np.abs(mu)
    if np.min(mu)<0:
        sys.exit('mu less than 0!')
    
    # Calculate g_loc using a regularization that approximates the next two lines.
    # g_loc = data.constants.secsYear*np.sqrt(pressure(H,data)/data.constants.rho)*(mu-data.param.muS)/(mu*data.param.b*data.param.d)
    # g_loc[g_loc<0] = 0
    
    #k = 50 # smoothing factor (small number equals more smoothing?)
   
    #f = [quad(calc_df,1e-4,x,args=(data.param.muS,k))[0] for x in mu] 
    #f = np.abs(f)
    #if np.min(f)<0:
    #    sys.exit('g_loc less than 0!')

    f = 1-data.param.muS/mu
    g_loc = data.constants.secsYear*np.sqrt(pressure(H,data)/data.constants.rho)*f/(data.param.b*data.param.d)
    #print(g_loc)
    g_loc[g_loc<0] = 0
    
    k = 100
    # Regularization of abs(mu-muS)
    f_mu = 2/k*np.logaddexp(0,k*(mu-data.param.muS)) - 2/k*np.logaddexp(0,-k*data.param.muS) + data.param.muS - mu
    

    zeta = f_mu/(data.param.A**2*data.param.d**2)
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
        # using linear interpolation to force g'=0 at X=L
        a_left[-1] = -0.5      
        a[-1] = 1.5
        
        # setting dg'/dx=0 at X=L; because we are using linear interpolation,
        # we don't need to worry about being on the grid vs on the staggered
        # grid (I don't think)
        a[0] = -1/(L*dx) 
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
def calc_muW(muW, H, W, U, data):
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
    
    _, _, u_mean = transverse(W, muW, H, data)
    
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
def transverse(W, muW, H, data): 
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

    y_c = W/2*(1-data.param.muS/muW) # critical value of y for which mu is no longer greater 
    # than muS; although flow occurs below this critical value, it is needed for 
    # computing g_loc (below)
       
    zeta = np.sqrt(np.abs(mu-data.param.muS))/(data.param.A*data.param.d)
    
    g_loc = np.zeros(len(y))
    g_loc[y<y_c] = data.constants.secsYear*np.sqrt(pressure(H,data)/data.constants.rho)*(mu[y<y_c]-data.param.muS)/(mu[y<y_c]*data.param.b*data.param.d) # local granular fluidity
    
    
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
def deformational_thickness(H,data):
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
    Hd = np.logaddexp(0,k*(H-data.param.d))/k
    
    return(Hd)
    

#%% calculate the effective pressure driving flow
def pressure(H,data):
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
    
    P = 0.5*data.constants.rho*data.constants.g*(1-data.constants.rho/data.constants.rho_w)*H
    
    return(P)


#%% thickness profile assuming quasi-static and constant mu_w and constant width

def quasistatic_thickness(H,data):
     
    mu_w = 0.6 # coefficient of friction along the fjord walls
    res = H - data.param.d*np.exp(mu_w*data.L*(1-data.x_)/data.W0 + (H-data.param.d)/(2*H))

    return(res)

#%%
def force(data):
    '''
    Calculate force per unit width acting on the glacier terminus

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    H = data.H0
    Hd = deformational_thickness(H,data)
    gg = data.gg[0] # not strictly at x=0 due to staggered grid, but we have set dg'/dx = 0 at x=0 so this is okay
    dUdx = (data.U[1]-data.U[0])/(data.dx*data.L)
    
    
    F = -2*H*pressure(Hd,data)*dUdx/gg + H*pressure(H,data)


    return(F)

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
    
    W_ = (data.W[:-1]+data.W[1:])/2
    H_ = (data.H[:-1]+data.H[1:])/2
    Hd_ = deformational_thickness(H_, data)
    muW_ = data.muW
    
    gg = np.concatenate(([1.5*data.gg[0]-0.5*data.gg[1]],data.gg,[1.5*data.gg[-1]-0.5*data.gg[-2]]))
    
    muW = np.concatenate(([3*data.muW[0]-3*data.muW[1]+data.muW[2]],data.muW,[3*data.muW[-1]-3*data.muW[-2]+data.muW[-3]]))
    muW[0] = np.min((muW[0], data.param.muW_max))
    muW[-1] = np.min((muW[-1], data.param.muW_max))

    
    ind = int(len(W)/2) # index of midpoint in fjord
    
    
    if data.subgrain_deformation=='n':
        y, u_transverse, _ = transverse(W_[ind], muW_[ind], Hd_[ind], data)

    else:
        y, u_transverse, _ = transverse((W[ind]+W[ind-1])/2, muW_[ind], (H[ind]+H[ind-1])/2, data)
        
    u_mean = np.trapz(u_transverse,y,y[1])/y[-1]
    #print(u_mean)
    #print(U[ind])    
    u_slip = U[ind+1]-u_mean#np.mean(u_transverse)
    u_transverse += u_slip

    ax1, ax2, ax3, ax4, ax5, ax_cbar = axes
    ax1.plot(X,U/data.constants.daysYear,color=plt.cm.viridis(color_id[k]))
    ax2.plot(np.append(X_,X_[::-1]),np.append(-data.constants.rho/data.constants.rho_w*H,(1-data.constants.rho/data.constants.rho_w)*H[::-1]),color=plt.cm.viridis(color_id[k]))
    ax3.plot(X_,gg,color=plt.cm.viridis(color_id[k]))
    ax4.plot(X,muW,color=plt.cm.viridis(color_id[k]))
    ax5.plot(np.append(y-y[-1],y),np.append(u_transverse,u_transverse[-1::-1])/data.constants.daysYear,color=plt.cm.viridis(color_id[k]))


    

    
