# try setting dH/dt and dL/dt = 0 to speed up solver?


import numpy as np

import os

from glaciome1D import glaciome, basic_figure, plot_basic_figure, constants

from scipy.integrate import trapz

import pickle

import time

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: 
    Jason Amundson
    University of Alaska Southeast
    jmamundson@alaska.edu
"""


# basic parameters needed for setting up the model; later will modify this so that 
# the fjord geometry can be passed through
constant = constants()

n_pts = 21 # number of grid points
L = 1e4 # ice melange length
Ut = 0.6e4 # glacier terminus velocity [m/a]; treated as a constant
Uc = 0.6e4 # glacier calving rate [m/a]; treated as a constant
Ht = 600 # terminus thickness
n = 101 # number of time steps
dt = 0.01# 1/(n_pts-1)/10 # time step [a]; needs to be quite small for this to work

# specifying fjord geometry
X_fjord = np.linspace(-200e3,200e3,101)
Wt = 4800
W_fjord = Wt + 0/10000*X_fjord


# set up basic figure
axes, color_id = basic_figure(n, dt)

data = glaciome(n_pts, dt, L, Ut, Uc, Ht, X_fjord, W_fjord)

data.diagnostic()
plot_basic_figure(data, axes, color_id, 0)
data.dt = 0.01
#data.steadystate()

j = 1
while j<50:
    print(j)
    data.prognostic()
    plot_basic_figure(data, axes, color_id, 50)
    j+=1

#%%
data.diagnostic()
plot_basic_figure(data, axes, color_id, 0)

for j in np.arange(1,50):
    print(j)
    data.prognostic()
    plot_basic_figure(data, axes, color_id, j)



#%%
start = time.time()
data.steadystate()
stop = time.time()
print(stop-start)

axes, color_id = basic_figure(n, dt)
plot_basic_figure(data, axes, color_id, 10)

data.B = 0
data.Uc = 0
data.steadystate()
plot_basic_figure(data, axes, color_id, 0)


#%%
axes, color_id = basic_figure(9, 0.01)
plot_basic_figure(data, axes, color_id, 0)
grid = np.linspace(31,101,8)

for j in np.arange(0,len(grid)):
    data.refine_grid(int(grid[j]))
    data.transient = 1
    k = 0
    while k<10:
        data.prognostic()
        k += 1
    data.transient = 0
    data.prognostic()
    plot_basic_figure(data, axes, color_id, j+1)


# #%%
# print('Solving diagnostic equations.')
# data.diagnostic()
# plot_basic_figure(data, axes, color_id, 0)

# #data.param.deps = 0.01 
# #data.diagnostic()
# #plot_basic_figure(data, axes, color_id, 100)



# #%%
# # run prognostic simulations
# start = time.time()
# L_old = data.L
# dL = 1000 # just initiating the change in length with some large value

# t = 0
# k = 0

# print('Solving prognostic equations.')

# #for k in np.arange(1,n):
# while np.abs(dL)>20:      
#     data.dt = 0.25*data.dx*data.L/np.max(data.U)
#     t += data.dt
#     k += 1
#     data.prognostic()
    
#     X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
#     H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))

    
     
#     if (k % 10) == 0:        
#         plot_basic_figure(data, axes, color_id, 0)
#         print('Time: ' + "{:.4f}".format(t) + ' years')   
#         print('Length: ' + "{:.2f}".format(data.L) + ' m')
#         print('Change in length: ' + "{:.2f}".format(data.L-L_old) + ' m') # over 10 time steps
#         print('Volume: ' + "{:.4f}".format(trapz(H, X_)*4000/1e9) + ' km^3')
#         print('H_L: ' + "{:.2f}".format(1.5*data.H[-1]-0.5*data.H[-2]) + ' m') 
#         print('CFL: ' + "{:.4f}".format(data.U[0]*data.dt/data.X[1]))
#         print(' ')
#         dL = data.L-L_old
#         L_old = data.L
#     # data.save(k)

    
# stop = time.time()

# print((stop-start)/60)           

# data.transient = 0
# data.prognostic()
# plot_basic_figure(data, axes, color_id, 100)

# #data.save('steady_B-0pt6_W' + str(Wt) + '_dwdx0.1.pickle')
# #%%
# data.refine_grid(21)

# start = time.time()
# L_old = data.L
# dL = 1000 # just initiating the change in length with some large value

# t = 0
# k = 0

# print('Solving prognostic equations.')

# #for k in np.arange(1,n):
# while np.abs(dL)>20:      
#     data.dt = 0.25*data.dx*data.L/np.max(data.U)
#     t += data.dt
#     k += 1
    
#     data.prognostic()
    
#     X_ = np.concatenate(([data.X[0]],data.X_,[data.X[-1]]))
#     H = np.concatenate(([data.H0],data.H,[1.5*data.H[-1]-0.5*data.H[-2]]))

    
     
#     if (k % 10) == 0:        
#         plot_basic_figure(data, axes, color_id, 0)
#         print('Time: ' + "{:.4f}".format(t) + ' years')   
#         print('Length: ' + "{:.2f}".format(data.L) + ' m')
#         print('Change in length: ' + "{:.2f}".format(data.L-L_old) + ' m') # over 10 time steps
#         print('Volume: ' + "{:.4f}".format(trapz(H, X_)*4000/1e9) + ' km^3')
#         print('H_L: ' + "{:.2f}".format(1.5*data.H[-1]-0.5*data.H[-2]) + ' m') 
#         print('CFL: ' + "{:.4f}".format(data.U[0]*data.dt/data.X[1]))
#         print(' ')
#         dL = data.L-L_old
#         L_old = data.L
#     # data.save(k)

    
# stop = time.time()

# print((stop-start)/60)           

# data.transient = 0
# data.prognostic()
# plot_basic_figure(data, axes, color_id, 100)
