# try setting dH/dt and dL/dt = 0 to speed up solver?


import numpy as np

import os

from glaciome1D import glaciome, basic_figure, plot_basic_figure

from scipy.integrate import trapz

import pickle

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: 
    Jason Amundson
    University of Alaska Southeast
    jmamundson@alaska.edu
"""


# COMMENTS
# 1. Have not yet included functionality for spatially variable melt rates
# 2. Have not yet included any parameterization of drag from the water
# 3. Write specific instructions for how to use this code

#%%
# basic parameters needed for setting up the model; later will modify this so that 
# the fjord geometry can be passed through

n_pts = 11 # number of grid points
L = 1e4 # ice melange length
Ut = 5e4 # glacier terminus velocity [m/a]; treated as a constant
Uc = 0*0.5e4 # glacier calving rate [m/a]; treated as a constant
Ht = 500 # terminus thickness
n = 101 # number of time steps
dt = 0.01 # time step [a]; needs to be quite small for this to work

# specifying fjord geometry
X_fjord = np.linspace(-200e3,200e3,101)
W_fjord = 4000*np.ones(len(X_fjord))

# Load spin-up or run spin-up if it hasn't already been done
if os.path.exists('spinup.pickle'):
    print('Loading spin-up file.')
    with open('spinup.pickle', 'rb') as file:
        data = pickle.load(file)
        file.close()
    data.dt = dt # update the time step size in the data model in case it has changed
else:
    print('Running model spin-up.')
    data = glaciome(n_pts, dt, L, Ut, Uc, Ht, X_fjord, W_fjord)
    #data.steadystate()
    
    # default is to assume no deformation below the grain scale
    # set data.subgrain_deformation = 'y' if you want to change this    
    data.subgrain_deformation = 'n'
    data.spinup()

# set up basic figure
axes, color_id = basic_figure(n, dt)

# plot spin-up results before running prognostic simulations
plot_basic_figure(data, axes, color_id, 0)



#%%
# run prognostic simulations

for k in np.arange(1,n):
    print('Time: ' + "{:.3f}".format(k*dt) + ' years')     
    
    data.prognostic()
    
    X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
    H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))
    
    print("{:.4f}".format(trapz(H, X_)*4000/1e9) + ' km^3')
    
     
    if (k % 1) == 0:        
        plot_basic_figure(data, axes, color_id, k)
        print(1.5*data.H[-1]-0.5*data.H[-2])  
    # data.save(k)
           
#%% temp
data.transient = 0
data.prognostic()
plot_basic_figure(data, axes, color_id, 0)
