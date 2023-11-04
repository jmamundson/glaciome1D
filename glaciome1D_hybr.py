# if having trouble with solver, switch from 'hybr' to 'lm'

import numpy as np
from scipy import sparse
from scipy.sparse import diags
from scipy.optimize import root, fsolve
from scipy.interpolate import interp1d
from scipy.integrate import simpson, cumtrapz, trapz

from matplotlib import pyplot as plt
import matplotlib

import sys

import pickle

import os
import time

#%%
class constants:
    
    def __init__(self):
        self.rho = 917.       # density of ice
        self.rho_w = 1028.    # density of water
        self.g = 9.81         # gravitational acceleration

        self.secsDay = 86400.
        self.daysYear = 365.25
        self.secsYear = self.secsDay*self.daysYear

constant = constants()


#%%
class parameters:
    '''
    Parameters for the nonlocal granular fluidity rheology
    '''
    
    def __init__(self):
        # you can probably decrease deps if you also decrease b
        self.deps = 0.1 # finite strain rate parameter [a^-1]; might work best to start with the large during spin up, then decrease it?
        self.d = 25 # characteristic iceberg size [m]
        self.A = 0.5 
        self.b = 1e4
        self.muS = 0.3
        self.muW_ = 0.6 # guess for muW iterations
        self.muW_max = 100 # maximum value for muW 
        
        self.Uscale = 5e3 # typical terminus velocity [5 km/yr]
        self.Lscale = 10e3 # typical ice melange length [10 km]
        self.Bscale = 100 # typical ice melange melt rate [100 m/yr]
        
        self.Hscale = self.Lscale*self.Bscale/self.Uscale # thickness scale [m]
        self.Tscale = self.Lscale/self.Uscale # time scale [yr]
        
        self.gamma = self.Hscale**2/self.Lscale**2
        
param = parameters()


#%%
class glaciome:
    '''
    The model class contains model variables, which simplifies passing variables
    into and out of functions, and methods for diagnostic, prognostic, and 
    steady-state solves.
    '''
    
    def __init__(self, n_pts, dt, L, Ut, Uc, Ht, X_fjord, W_fjord):
        '''
        Initialize an object at t=0.

        Parameters
        ----------
        n_pts : number of grid points
        dt : time step [a]
        L : ice melange length [m]
        Ut : terminus velocity [m/a]
        Uc : calving rate [m/a]
        Ht = terminus thickness
        X_fjord = coordinates for defining the fjord geometry [m]
        W_fjord = fjord width at X_fjord [m]
        '''
        
        # import the constants into the glaciome class for record keeping
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

        # create width interpolator and find width at initial grid points
        self.X_fjord = X_fjord
        self.W_fjord = W_fjord        
        self.width_interpolator = interp1d(self.X_fjord, self.W_fjord, fill_value='extrapolate') #self.__create_width_interpolator(X_fjord, W_fjord)
        self.W = np.array([self.width_interpolator(x) for x in self.X_])
        self.W0 = self.width_interpolator(self.X[0])
        self.WL = self.width_interpolator(self.X[-1])

        # use the quasistatic thickness (Amundson and Burton, 2018) for the 
        # initial thickness profile
        self.H = fsolve(self.__quasistatic_thickness, 2*self.param.d*np.ones(len(self.x_)))
        self.H0 = 1.5*self.H[0]-0.5*self.H[1]
        self.HL = self.param.d
        
        # specify glacier terminus thickness, velocity, and calving rate, and 
        # use those values to determine the initial ice melange velocity (assumed
        # constant)
        self.Ht = Ht
        self.Ut = Ut
        self.Uc = Uc
        self.U0 = self.Uc*self.Ht/self.H0 # !!! self.Ut + self.Uc*(self.Ht/self.H0-1)
        self.U = self.U0*np.ones(len(self.x))
        
        # specify some initial values for muW and gg; these will be adjusted
        # during the first diagnostic solve
        self.muW = self.param.muW_*np.ones(len(self.U))
        self.gg = self.param.deps/self.muW[0]*np.ones(len(self.H))
        
        # set the specific mass balance rate (treated as spatially constant)
        self.B = -0.8*self.constants.daysYear 
        
        # set time step and initial time
        self.dt = dt
        self.t = 0 

        self.tauX = 0 # drag force N/m; later divide by L to cast in terms of a stress; note units have 1/s^2, but it will be divided by gravity so it's okay
        
        self.transient = 1 # 1 for transient simulation, 0 to try to solve for steady-state
        
    
    def nondimensionalize(self):
        self.H = self.H/self.param.Hscale
        self.H0 = self.H0/self.param.Hscale
        self.HL = self.HL/self.param.Hscale
        self.Ht = self.Ht/self.param.Hscale
        self.param.d = self.param.d/self.param.Hscale
        
        self.U = self.U/self.param.Uscale
        self.Ut = self.Ut/self.param.Uscale
        self.Uc = self.Uc/self.param.Uscale
        self.U0 = self.U0/self.param.Uscale
        
        self.L = self.L/self.param.Lscale
        self.W = self.W/self.param.Lscale
        self.W0 = self.W0/self.param.Lscale
        self.WL = self.WL/self.param.Lscale
        self.width_interpolator = interp1d(self.X_fjord/self.param.Lscale, self.W_fjord/self.param.Lscale, fill_value='extrapolate')
        
        self.B = self.B/self.param.Bscale
        
        self.dt = self.dt/self.param.Tscale
        
        self.gg = self.gg*self.param.Lscale/self.param.Uscale
    
    
    def redimensionalize(self):
        self.H = self.H*self.param.Hscale
        self.H0 = self.H0*self.param.Hscale
        self.HL = self.HL*self.param.Hscale
        self.Ht = self.Ht*self.param.Hscale
        self.param.d = self.param.d*self.param.Hscale
        
        self.U = self.U*self.param.Uscale
        self.Ut = self.Ut*self.param.Uscale
        self.Uc = self.Uc*self.param.Uscale
        self.U0 = self.U0*self.param.Uscale
        
        self.L = self.L*self.param.Lscale
        self.W = self.W*self.param.Lscale
        self.W0 = self.W0*self.param.Lscale
        self.WL = self.WL*self.param.Lscale
        self.width_interpolator = interp1d(self.X_fjord, self.W_fjord, fill_value='extrapolate')

        
        
        self.B = self.B*self.param.Bscale
        
        self.dt = self.dt*self.param.Tscale
        
        self.gg = self.gg*self.param.Uscale/self.param.Lscale
        
        
    
    def diagnostic(self):
        '''
        Solve for the velocity, granular fluidity, and muW for a given model geometry.
        '''
        
        self.nondimensionalize()
        
        UggmuW = np.concatenate((self.U,self.gg,self.muW)) # starting point for solving the differential equations
        # result = root(self.__solve_diagnostic, UggmuW, method='hybr', options={'maxfev':int(1e6)})
        result = root(self.__solve_diagnostic, UggmuW, method='lm', options={'maxiter':int(1e6)})
        
        if result.success != 1:
            print('status: ' + str(result.status))
            print('success: ' + str(result.success))
            print('message: ' + result.message)
            print('')
        
        self.U = result.x[:len(self.x)]
        self.gg = result.x[len(self.x):2*len(self.x)-1]
        self.muW = result.x[2*len(self.x)-1:]
        
        self.redimensionalize()
        
    
    def prognostic(self):
        '''
        Use an implicit time step to determine the velocity, granular fluidity, 
        muW, thickness, and length.
        '''
        
        self.nondimensionalize()
        
        # The previous thickness and length are required.
        H_prev = self.H
        L_prev = self.L
        
        self.X[0] += (self.Ut-self.Uc)*self.dt # !!! use an explicit time step to find new position X0
        
        UggmuWHL = np.concatenate((self.U,self.gg,self.muW,self.H,[self.L])) # starting point for solving the differential equations
        result = root(self.__solve_prognostic, UggmuWHL, (H_prev, L_prev), method='hybr', options={'maxfev':int(1e6), 'xtol':1e-9})
        # result = root(self.__solve_prognostic, UggmuWHL, (H_prev, L_prev), method='lm', options={'maxiter':int(1e6)})
        
        if result.success != 1:
            print('status: ' + str(result.status))
            print('success: ' + str(result.success))
            print('message: ' + result.message)
            print('')
        
        self.U = result.x[:len(self.x)]
        self.gg = result.x[len(self.x):2*len(self.x)-1]
        self.muW = result.x[2*len(self.x)-1:3*len(self.x)-1]
        self.H = result.x[3*len(self.x)-1:-1]
        self.L = result.x[-1]
        
        self.X = self.X*self.param.Lscale
        self.X_ = self.X_*self.param.Lscale
        
        # Update the time stored within the model object.
        self.t += self.dt
        
        self.redimensionalize()
        
        
    def refine_grid(self, n_pts):
        '''
        It is sometimes helpful to spin up the model with a coarse grid (e.g., 
        n_pts = 21), refine the grid, and then run to a new steady-state.
        '''
        
        x = np.linspace(0,1,n_pts)
        x_ = (x[:-1]+x[1:])/2
        
        X = x*self.L + self.X[0]
        X_ = x_*self.L + self.X[0]
        
        U_interpolator = interp1d(self.X, self.U, kind='linear', axis=-1, fill_value='extrapolate')
        H_interpolator = interp1d(self.X_, self.H, kind='linear', axis=-1, fill_value='extrapolate')
        W_interpolator = interp1d(self.X_, self.W, kind='linear', axis=-1, fill_value='extrapolate')
        gg_interpolator = interp1d(self.X_, self.gg, kind='linear', axis=-1, fill_value='extrapolate')
        muW_interpolator = interp1d(self.X, self.muW, kind='linear', axis=-1, fill_value='extrapolate')
        
        self.U = U_interpolator(X)
        self.H = H_interpolator(X_)
        self.W = W_interpolator(X_)
        self.gg = gg_interpolator(X_)
        self.muW = muW_interpolator(X)
        
        self.dx = x[1]
        self.x = x
        self.x_ = x_
        self.X = X
        self.X_ = X_
            
    
    def steadystate(self):
        print('Solving diagnostic equation.')
        start = time.time()
        self.diagnostic() # solve diagnostic equation first to ensure that model is consistent
        stop = time.time()
        print('Diagnostic solve took ' + "{:.02f}".format(stop-start) + ' s')
        L_old = self.L
        
        k = 1 # initiate counter
        t = 0
        t_old = 0
        dLdt = 1000 # initializing with some large value
        
        # axes, color_id = basic_figure(10000, 0.01)
        # plot_basic_figure(self, axes, color_id, 0)

        t_step_old = time.time()
        
        X_ = np.concatenate(([self.X[0]], self.X_, [self.X[-1]]))
        H = np.concatenate(([self.H0], self.H, [self.HL]))
        W = np.concatenate(([self.W0], self.W, [self.WL]))
        # V_old = simpson(H*W, X_)*1e-9
        V_old = simpson(self.H[1:-1]*self.W[1:-1], self.X_[1:-1])*1e-9
        #dVdt2_old = (self.U0*self.H0*self.W0 + self.B*simpson(W,X_) - self.U[-1]*self.param.d*self.WL)*1e-9
        
        print('Solving prognostic equations.')
        while np.abs(np.abs(dLdt))>10: # !!! may need to adjust if final solution looks noisy      
            self.dt = 0.1*self.dx*self.L/np.max(self.U) # use an adaptive time step, loosely based on CFL condition (for explicit schemes)
            
            self.prognostic()
            t += self.dt
            
            X_ = np.concatenate(([self.X[0]], self.X_, [self.X[-1]]))
            H = np.concatenate(([self.H0], self.H, [self.HL]))
            W = np.concatenate(([self.W0], self.W, [self.WL]))
            # V = simpson(H*W, X_)*1e-9
            
            V = simpson(self.H[1:-1]*self.W[1:-1], self.X_[1:-1])*1e-9
            
            dVdt1 = (V-V_old)/self.dt
            
            dVdt2 = (self.U0*self.H0*self.W0 + self.B*simpson(W,X_) - self.U[-1]*self.param.d*self.WL)*1e-9
            
            dVdt2 = ((self.U[0]+self.U[1])/2*self.H[0]*self.W[0] + self.B*simpson(self.W,self.X_) - (self.U[-1]+self.U[-2])/2*self.H[-1]*self.W[-1])*1e-9
            
            if (k%10) == 0:
                
                
                t_step = time.time()
                
                print('Step: ' + str(int(k)) )
                print('Time per step: ' + "{:.2f}".format((t_step-t_step_old)/10) + ' s')
                print('Simulation time: ' + "{:.3f}".format(t) + ' yr')
                print('Length: ' + "{:.2f}".format(self.L) + ' m')
                print('dL/dt: ' + "{:.2f}".format((self.L-L_old)/(t-t_old)) + ' m/yr') # over 10 time steps
                print('Volume: ' + "{:.4f}".format(V) + ' km^3')
                print('dV/dt: ' + "{:.2f}".format(dVdt1) + ' km^3/yr' )
                # print('dV/dt: ' + "{:.2f}".format(self.W[0]*(self.U0*self.H0+self.B*self.L-self.U[-1]*self.param.d)*1e-9) + ' km^3/yr')
                print('dV/dt: ' + "{:.2f}".format(dVdt2) + ' km^3/yr')
                #print('dV/dt: ' + "{:.2f}".format(dVdt2_old) + ' km^3/yr')

                print('H_L: ' + "{:.2f}".format(1.5*self.H[-1]-0.5*self.H[-2]) + ' m') 
                print('CFL: ' + "{:.4f}".format(np.max(self.U)*self.dt/(self.dx*self.L)))
                print(' ')
                dLdt = (self.L-L_old)/(t-t_old)
                L_old = self.L
                
                
                t_old = t
                t_step_old = t_step
                
                
                # plot_basic_figure(self, axes, color_id, k)
           
            V_old = V
            
            k += 1
        
        self.transient = 0
        print('Steady-state solve.')
        self.prognostic()
        # plot_basic_figure(self, axes, color_id, 999)
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
    
    
    def __solve_diagnostic(self,UggmuW): 
        '''
        This function is minimized in order to determine the velocity, granular
        fluidity, and muW

        Parameters
        ----------
        UggmuW : an array containing the initial guess for the velocity, granular
        fluidity, and muW

        Returns
        -------
        res : residual of differential equations
        '''
        
        # extract current values of U, gg, and muW
        self.U = UggmuW[:len(self.x)]
        self.gg = UggmuW[len(self.x):2*len(self.x)-1]
        self.muW = UggmuW[2*len(self.x)-1:]
        # !!! self.muW[self.muW > param.muW_max] = param.muW_max
        
        # compute residuals of velocity and granular fluidity differential equations
        resU = self.__calc_U(self.U)
        resgg = self.__calc_gg(self.gg)
        
        # for each iteration, muW is used to produce a transverse velocity 
        # profile and then compare the average velocity of the profile to U
        # !!! this is a hack so that I don't have to modify the transverse equations just yet
        self.redimensionalize()
        tmp = [self.transverse(x) for x in self.x]
        Ubar = np.array([tmp[j][2] for j in range(len(tmp))]) / self.param.Uscale
        self.nondimensionalize()
        # !!! end temporary hack
        
        resmuW = (Ubar - (self.U+self.Ut))
        
        # combine the residuals into a single variable to be minimized
        res = np.concatenate((resU,resgg,resmuW))
      
        return(res)

    
    def __solve_prognostic(self, UggmuWHL, H_prev, L_prev):
        '''
        Computes the residual for the velocity, granular fluidity, thickness, and
        length differential equations. The residual is minimized in model.implicit.
        '''
        
        # extract current value for U, gg, H, L
        self.U = UggmuWHL[:len(self.x)]
        self.gg = UggmuWHL[len(self.x):2*len(self.x)-1]
        self.muW = UggmuWHL[2*len(self.x)-1:3*len(self.x)-1]
        self.muW[self.muW > param.muW_max] = param.muW_max        
        
        self.H = UggmuWHL[3*len(self.x)-1:-1]
        self.H0 = 1.5*self.H[0]-0.5*self.H[1]
        self.HL = 1.5*self.H[-1]-0.5*self.H[-2]
        self.U0 = self.Uc*self.Ht/self.H0
        self.L = UggmuWHL[-1]
    
        self.dLdt = self.transient*(self.L-L_prev)/self.dt
    
        # Update the dimensional grid and the width
        self.X = self.x*self.L # !!! + self.X[0]
        self.X_ = (self.X[:-1]+self.X[1:])/2
        self.W = self.width_interpolator(self.X_) 
    
        # update endpoints, since W and H are on the staggered grid
        self.W0 = self.width_interpolator(self.X[0]) 
        self.WL = self.width_interpolator(self.X[-1]) 
        self.H0 = 1.5*self.H[0]-0.5*self.H[1]
        self.HL = 1.5*self.H[-1]-0.5*self.H[-2]
        
        resU = self.__calc_U(self.U) 
        resgg = self.__calc_gg(self.gg) 
        
        # for each iteration, muW is used to produce a transverse velocity 
        # profile and then compare the average velocity of the profile to U
        # !!! temporary hack
        self.redimensionalize()
        tmp = [self.transverse(x) for x in self.x]
        Ubar = np.array([tmp[j][2] for j in range(len(tmp))]) / self.param.Uscale
        self.nondimensionalize()
        # !!! end temporary hack
        
        resmuW = (Ubar - (self.U+self.Ut)) 
        
        resH = self.__calc_H(self.H, H_prev) 
        
        resHc = (1.5*self.H[-1]-0.5*self.H[-2]-self.param.d)
        
        # append residuals into single variable to be minimized
        resUggmuWHL = np.concatenate((resU, resgg, resmuW, resH, [resHc]))    
        
        
        return(resUggmuWHL)


    def __calc_U(self,U):
        '''
        Compute the residual for velocity differential equation
        
        Parameters
        ----------
        U : initial guess for U
        
        Returns
        -------
        res : residual of the velocity differential equation
    
        '''
        
        # extract variables from the model object
        H = self.H
        gg = self.gg
        W = self.W
        dx = self.dx
        L = self.L
        muW = self.muW[1:-1]
        U0 = self.U0
            
        # determine H and W on the grid in order to calculate the coefficient of friction along the fjord walls
        # note that these start at the second grid point
        H_ = (H[:-1]+H[1:])/2 # thickness on the grid, excluding the first and last grid points
        W_ = (W[:-1]+W[1:])/2 # width on the grid, excluding the first and last grid points        
        
        
        option = '1'
        
        if (option == '1') or (option == '2'):
            nu = H**2/gg
            
        elif option == '3':
            exp = 1 # changing exp has only modest impacts
            nu = H**2/gg / (1-(param.d/H)**exp) # option 3

        
        a_left = nu[:-1]/(dx*L)**2    
        a_left = np.append(a_left,-1/(dx*L))
        
        a = np.ones(len(U))/(dx*L)
        a[1:-1] = -(nu[1:] + nu[:-1])/(dx*L)**2
        a[0] = 1
        
        a_right = nu[1:]/(dx*L)**2
        a_right = np.append(0,a_right)
        
        diagonals = [a_left,a,a_right]
        D = diags(diagonals,[-1,0,1]).toarray() 
        
        
        if option == '1':
            T = H_*(H[1:]-H[:-1])/(dx*L) + H_**2/W_*muW*np.sign(U[1:-1]) - (self.tauX/self.L)/(constant.rho*constant.g*(1-constant.rho/constant.rho_w))

        elif option == '2':
            T = (H_-self.param.d/2)*(H[1:]-H[:-1])/(dx*L) + H_**2/W_*muW*np.sign(U[1:-1]) - (self.tauX/self.L)/(constant.rho*constant.g*(1-constant.rho/constant.rho_w))

        elif option == '3':
            T = H_*(H[1:]-H[:-1])/(dx*L) + H_**2/W_*muW*np.sign(U[1:-1]) - (self.tauX/self.L)/(constant.rho*constant.g*(1-constant.rho/constant.rho_w)) 
            
        # upstream boundary condition
        T = np.append(U0,T)
        
        # downstream boundary condition; longitudinal resistive stress equals
        # difference between glaciostatic and hydrostatic stresses
        
        if option == '1':
            T = np.append(T,0*0.5*self.gg[-1])
            
        elif option == '2':
            T = np.append(T,0.5*self.gg[-1] * (1 - (self.param.d/self.HL))) 
        
        elif option == '3':
            T = np.append(T,0.5*self.gg[-1] * (1 - (self.param.d/self.HL)**exp)) 
        
        
        res = np.matmul(D,U)-T
        
        return(res)
    

    def __calc_gg(self,gg):
        '''
        Compute the residual for the granular fluidity differential equation.
        
        Parameters
        ----------
        gg : initial guess for the granular fluidity
        
        Returns
        -------
        res : residual of the granular fluidity differential equation
        '''
        
        # extract variables from the model object
        H = self.H
        L = self.L
        dx = self.dx
        
        
        # calculate current values of ee_chi, mu, g_loc, and zeta
    
        # second invariant of the strain rate in transformed coordinate system
        ee_chi = self.__second_invariant()
        
        mu = (ee_chi/L+param.deps*self.param.Lscale/self.param.Uscale)/gg # option 1
        #mu = np.sqrt((ee_chi/L)**2+param.deps**2)/gg # option 2 !!! seems best?
        #mu = (ee_chi/L)/(1-np.exp(-(ee_chi/L)/param.deps)) / gg # option 3
        
        
        mu = np.abs(mu)
        if np.min(mu)<0:
            sys.exit('mu less than 0!')
        
        
        f = 1-self.param.muS/mu
        g_loc = constant.secsYear*np.sqrt(self.pressure(H)/(constant.rho*self.param.d**2*self.param.Hscale))*f/(self.param.b)
        g_loc[g_loc<0] = 0
        
        k = 100
        # Regularization of abs(mu-muS)
        f_mu = 2/k*np.logaddexp(0,k*(mu-self.param.muS)) - 2/k*np.logaddexp(0,-k*self.param.muS) + self.param.muS - mu
        
    
        zeta = f_mu/(self.param.A**2*self.param.d**2)
        # Essentially Equation 19 in Amundson and Burton (2018)
        
        # construct equation Dx=T
        # boundary conditions:
        #    dg/dx=0 is the soft boundary condition recommended by Henann and Kamrin (2013)
        
        
        a_left = self.param.gamma*np.ones(len(mu)-1)/(L*dx)**2
        a = -(self.param.gamma*2/(L*dx)**2 + zeta)    
        a_right = self.param.gamma*np.ones(len(mu)-1)/(L*dx)**2 
           
    
        # using linear interpolation to force g'=0 at X=L
        # a_left[-1] = -0.5/param.gscale      
        # a[-1] = 1.5/param.gscale
        
        # setting dg'/dx = 0 at X=L         !!! option 1, 2, and 3
        a_left[-1] = -1/(L*dx)
        a[-1] = 1/(L*dx)
        
        # setting dg'/dx=0 at X=L; because we are using linear interpolation,
        # we don't need to worry about being on the grid vs on the staggered
        # grid (I don't think)
        a[0] = -1/(L*dx) 
        a_right[0] = 1/(L*dx)
    
        diagonals = [a_left,a,a_right]
        D = diags(diagonals,[-1,0,1]).toarray() 
        
        T = -zeta*g_loc
    
        T[0] = 0 
        T[-1] = 0
    
        res = np.matmul(D,gg) - T
    
        return(res)


    def __calc_H(self, H, H_prev):    
        '''
        Compute the residual for the thickness differential equation.
    
        Parameters
        ----------
        H : initial guess for the thickness
        H_prev : thickness profile from the previous time step
        
        Returns
        -------
        res : residual for the thickness differential equation
        '''
        
        # extract variables from the model object
        x_ = self.x_
        dx = self.dx
        dt = self.dt
        W = self.W
        L = self.L
        U = self.U
        B = self.B
        Ut = self.Ut
        Uc = self.Uc
        dLdt = self.dLdt
        
        # defined for simplicity later  
        beta = x_*dLdt
        
        a_left = 1/(2*dx*L)*(beta[1:] - W[:-1]/W[1:]*(U[1:-1]+U[:-2]))
        a_left[-1] = 1/(dx*L)*(beta[-1]-0.5*W[-2]/W[-1]*(U[-2]+U[-3]))
        
        a = self.transient*1/dt + 1/(2*dx*L)*(U[2:]+U[1:-1])
        a[-1] = self.transient*1/dt+1/(dx*L)*(-beta[-1]+0.5*(U[-1]+U[-2])) 
        a = np.append(self.transient*1/dt + 1/(dx*L)*(beta[0] - (U[0]+U[1])/2) , a)
        
        a_right = -1/(2*dx*L)*beta[:-1]
        a_right[0] = 1/(L*dx)*(-beta[0]+(U[2]+U[1])/2*W[1]/W[0])
        
        T = B + self.transient*H_prev/dt
        
        
        diagonals = [a_left,a,a_right]
        D = diags(diagonals,[-1,0,1]).toarray()
        
        # set d^2{H}/dx^2=0 across the first three grid points (to deal with upwind scheme)
        D[0,0] = 1
        D[0,1] = -2
        D[0,2] = 1
        T[0] = 0
        
        res = np.matmul(D,H) - T
    
        return(res)


    def transverse(self, x): 
        '''
        Calculates transverse velocity profiles for the nonlocal granular fluidity
        rheology at dimensionless coordinate x. See Amundson and Burton (2018) for details.
    
        Parameters
        ----------
        x: location of the transect, in dimensionless coordinates
        
        Returns
        -------
        y : transverse coordinate [m]
        u : velocity at y, assuming no slip along the boundary [m a^-1]
        u_mean : mean velocity across the profile, assuming no slip along the boundary [m a^-1]
    
        '''
        
        # extract W, muW, and H at the location of interest
        W = np.interp(x, self.x_, self.W)
        muW = np.interp(x, self.x, self.muW)
        H = np.interp(x, np.concatenate(([0],self.x_,[1])), np.concatenate(([self.H0],self.H,[self.HL])))
        
        n_pts = 101 # number of points in half-width
        y = np.linspace(0,W/2,n_pts) # location of points
        
        dy = y[1] # grid spacing
        
        mu = muW*(1-2*y/W) # variation in mu across the fjord, assuming quasi-static flow
    
        y_c = W/2*(1-self.param.muS/muW) # critical value of y for which mu is no longer greater 
        # than muS; although flow occurs below this critical value, it is needed for 
        # computing g_loc (below)
           
        zeta = np.sqrt(np.abs(mu-self.param.muS))/(self.param.A*self.param.d)
        
        g_loc = np.zeros(len(y))
        g_loc[y<y_c] = constant.secsYear*np.sqrt(self.pressure(H)/(constant.rho*self.param.d**2))*(mu[y<y_c]-self.param.muS)/(mu[y<y_c]*self.param.b) # local granular fluidity
        g_loc = g_loc*self.param.Lscale/self.param.Uscale
        
        # Compute residual for the granular fluidity. we set dg/dy = 0 at
        # y = 0 and at y = W/2. Because mu is known, this does not need to be done
        # iteratively (as it is done in the along-flow direction.)    
        
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
        
        
        u = cumtrapz(2*mu*gg, y, initial=0)
        u_mean = 2/W*trapz(u,y)
        
        
        # transform double integral using volterra integral equation
        u_mean = 2/W*trapz(2*mu*gg*(y[-1]-y),y)
        
        return(y,u,u_mean)


    def __second_invariant(self):
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
        
        ee_chi = np.sqrt((np.diff(self.U)/self.dx)**2/2)

        return(ee_chi)


    def __quasistatic_thickness(self,H):
        '''
        Used for determining the thickness profile assuming quasi-static flow, constant mu_w, and constant width
        
        Parameters
        ----------
        H : initial guess for the thickness profile [m]

        Returns
        -------
        res : need to determine H iteratively in order to solve implicit equation
        '''
    
        mu_w = param.muW_ # coefficient of friction along the fjord walls
        res = H - param.d*np.exp(mu_w*self.L*(1-self.x_)/self.W0 + (H-param.d)/(2*H))
    
        return(res)


    def force(self):
        '''
        Calculate force per unit width acting on the glacier terminus.
        '''
        
        H0 = self.H0
        gg = self.gg[0] # not strictly at x=0 due to staggered grid, but we have set dg'/dx = 0 at x=0 so this is okay
        dUdx = (self.U[1]-self.U[0])/(self.dx*self.L)
        
        F = -2*H0*self.pressure(H0)*dUdx/gg + H0*self.pressure(H0)
    
        return(F)


    def pressure(self,H):
        '''
        Calculate the difference between the depth-averaged glaciostatic and 
        hydrostatic pressures, assuming at flotation.    
    
        Parameters
        ----------
        H : thickness [m]
        
        Returns
        -------
        P : pressure [Pa]
        '''
        
        P = 0.5*constant.rho*constant.g*(1-constant.rho/constant.rho_w)*H
        
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
    
    
    xmax = 30000
    vmax = 200
    
    ax1 = plt.axes([left, bot+ax_height+2.25*ygap, ax_width, ax_height])
    ax1.set_xlabel('Longitudinal coordinate [m]')
    ax1.set_ylabel('Speed [m/d]')
    ax1.set_ylim([0,vmax])
    ax1.set_xlim([0,xmax])
    
    ax2 = plt.axes([left+ax_width+xgap, bot+ax_height+2.25*ygap, ax_width, ax_height])
    ax2.set_xlabel('Longitudinal coordinate [m]')
    ax2.set_ylabel('Elevation [m]')
    ax2.set_ylim([-500, 100])
    ax2.set_xlim([0,xmax])
    
    ax3 = plt.axes([left, bot+1.25*ygap, ax_width, ax_height])
    ax3.set_xlabel('Longitudinal coordinate [m]')
    ax3.set_ylabel('$g^\prime$ [a$^{-1}]$')
    ax3.set_ylim([0, 1])
    ax3.set_xlim([0,xmax])
    
    ax4 = plt.axes([left+ax_width+xgap, bot+1.25*ygap, ax_width, ax_height])
    ax4.set_xlabel('Longitudinal coordinate [m]')
    ax4.set_ylabel('$\mu_w$')
    ax4.set_ylim([0, 1])
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
    U = data.U + data.Ut
    H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))
    muW = data.muW
    
    gg = np.concatenate(([1.5*data.gg[0]-0.5*data.gg[1]],data.gg,[1.5*data.gg[-1]-0.5*data.gg[-2]]))
    
    y, u_transverse, u_mean = data.transverse(0.5)
    # U_ind = np.interp(0.5,data.x,U)
    
    # u_slip = U_ind-u_mean
    # u_transverse += u_slip

    ax1, ax2, ax3, ax4, ax5, ax_cbar = axes
    ax1.plot(X,U/data.constants.daysYear,color=plt.cm.viridis(color_id[k]))
    ax2.plot(np.append(X_,X_[::-1]),np.append(-data.constants.rho/data.constants.rho_w*H,(1-data.constants.rho/data.constants.rho_w)*H[::-1]),color=plt.cm.viridis(color_id[k]))
    ax3.plot(X_,gg,color=plt.cm.viridis(color_id[k]))
    ax4.plot(X,muW,color=plt.cm.viridis(color_id[k]))
    ax5.plot(np.append(y-y[-1],y),np.append(u_transverse,u_transverse[-1::-1])/data.constants.daysYear,color=plt.cm.viridis(color_id[k]))