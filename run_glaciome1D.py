import numpy as np

import os

from glaciome1D import model, basic_figure, plot_basic_figure

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
# 1. currently only able to handle constant width
# 2. dH/dt upstream boundary condition
# 3. calving of ice from melange


#%%
n_pts = 11 # number of grid points
L = 1e4 # ice melange length
Ut = 1e4 # glacier terminus velocity [m/a]; treated as a constant
W = 4000 # ice melange width [m], treated as a constant
n = 101 # number of time steps
dt = 0.01 # time step [a]; needs to be quite small for this to work


# Load spin-up or run spin-up if it hasn't already been done
if os.path.exists('spinup.pickle'):
    print('Loading spin-up file.')
    with open('spinup.pickle', 'rb') as file:
        data = pickle.load(file)
        file.close()
    data.dt = dt # update the time step size in the data model in case it has changed
else:
    print('Running model spin-up.')
    data = model(n_pts, dt, L, Ut, W)
    data.spinup()

# set up basic figure
axes, color_id = basic_figure(n, dt)

# plot spin-up results before running prognostic simulations
plot_basic_figure(data, axes, color_id, 0)





#%% start prognostic simulations
Ugg = np.concatenate((data.U,data.gg))

for k in np.arange(1,n):
    print('Time: ' + "{:.2f}".format(k*dt) + ' years')     
    
    # choose between explicit or implicit time steps
    data.explicit() 
    #data.implicit()
    
    if (k % 10) == 0:        
        plot_basic_figure(data, axes, color_id, k)
            
           


#plt.savefig('test.png',format='png',dpi=150)    