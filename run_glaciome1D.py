import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from scipy.optimize import root
from scipy.optimize import newton_krylov
from scipy.optimize import fsolve

import config

import importlib

import os

from granular_fluidity import model
from granular_fluidity import basic_figure, plot_basic_figure

import pickle

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jason
"""


# COMMENTS
# 1. currently only able to handle constant width
# 2. dH/dt upstream boundary condition
# 3. calving of ice from melange
# 4. need to be careful with handling of dt within object, not working correctly yet.

#%%
n_pts = 11
L = 1e4
Ut = 1e4
W = 4000
n = 5 # number of time steps
dt = 0.01 # time step [a]; needs to be quite small for this to work


# Load spin-up or run spin-up if it hasn't already been done
if os.path.exists('spinup.pickle'):
    print('Loading spin-up file.')
    with open('spinup.pickle', 'rb') as file:
        data = pickle.load(file)
        file.close()
else:
    print('Running model spin-up.')
    data = model(n_pts, dt, L, Ut, W)
    data.spinup()

data.dt = dt
# set up basic figure
axes, color_id = basic_figure(n, dt)

# plot spin-up results before running prognostic simulations
plot_basic_figure(data, axes, color_id, 0)





#%% start prognostic simulations
Ugg = np.concatenate((data.U,data.gg))

for k in np.arange(1,n):
    print('Time: ' + "{:.2f}".format(k*dt) + ' years')     
    
    data.explicit() 
    #data.implicit()
    
    if (k % 1) == 0:        
        plot_basic_figure(data, axes, color_id, k)
            
           
#ax4.plot(np.array([0,1e4]),np.array([config.muS,config.muS]),'k:')

#plt.savefig('test.png',format='png',dpi=150)    