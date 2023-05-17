import numpy as np
from matplotlib import pyplot as plt
import matplotlib

from scipy.optimize import root
from scipy.optimize import newton_krylov
from scipy.optimize import fsolve

from general_utilities import second_invariant, width, pressure

import config

import importlib

# here you should specify the rheology that should be imported
rheology = 'granular_fluidity'
model = importlib.import_module(rheology)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: jason
"""


#%%

x0 = 0 # initial left boundary of the melange [m]
L = 10000 # initial ice melange length [m]
dx = 0.1  # grid spacing (in dimensionless units)
x = np.arange(0,1+dx,dx) # longitudinal grid
x_ = (x[:-1]+x[1:])/2
X = x*L # coordinates of unstretched grid
X_ = x_*2 # coordinates of staggered grid

W = width((X[:-1]+X[1:])/2) # later, W needs be modified to change with the grid

H = np.ones(len(x)-1)*config.d # initial ice melange thickness [m]

Ut = 10000 # width-averaged glacier terminus velocity [m/yr]

U = Ut*np.exp(-x/0.5) # initial guess for the velocity
gg = 1*(1-x[:-1]) # initial guess for the granular fluidity on staggered grid

B = -0/config.secsYear # mass balance rate [m s^-1]; treating as constant for now



n = 21 # number of time steps
color_id = np.linspace(0,1,n+1) # for cycling through colors in a plot 
dt = 0.025 # time step [yr]


residual = Ut*np.ones(U.shape) # just some large values for the initial residuals
j = 0

# alternate solving for granular fluidity and velocity in order to get a good starting point
print('Iterating between granular fluidity and velocity in order to find a good starting point')
while np.max(np.abs(residual))>1:
    print('Velocity residual: ' + "{:.2f}".format(np.max(np.abs(residual))) + ' m/a')
    result = root(model.calc_gg, gg, (U,H,L,dx), method='lm')
    gg = result.x
    
    result = root(model.calc_U, U, (gg, x, X, Ut, H, W, dx, L), method='lm')
    U_new = result.x
    
    residual = U-U_new
    
    U = U_new
    j += 1    

print('Done with initial iterations')

# now simultaneous solve for velocity and granular fluidity
Ugg = np.concatenate((U,gg))
result = root(model.diagnostic, Ugg, (x,X,Ut,H,W,dx,L), method='lm', tol=1e-6)

U = result.x[:len(x)]
gg = result.x[len(x):]


#np.savez('spinup.npz',x=x,X=X,X_=X_,dx=dx,Ut=Ut,U=U,H=H,W=W,L=L,gg=gg,B=B)

#%% set up figure and plot initial conditions
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


muW = model.get_muW(x,U,H,W,L,dx)
W = width((X[:-1]+X[1:])/2)
ind = int(len(W)/2) # index of midpoint in fjord
y, u_transverse, _ = model.transverse(W[ind],muW[ind],H[ind])

ax1.plot(X,U/config.daysYear,color=plt.cm.viridis(color_id[0]))
ax2.plot(np.append(X_,X_[::-1]),np.append(-config.rho/config.rho_w*H,(1-config.rho/config.rho_w)*H[::-1]),color=plt.cm.viridis(color_id[0]))
ax3.plot(X_,gg,color=plt.cm.viridis(color_id[0]))
ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[0]))
ax5.plot(np.append(y,y+y[-1]),np.append(u_transverse,u_transverse[-1::-1])/config.daysYear,color=plt.cm.viridis(color_id[0]))





#%%
# things to try adjusting: 
# 1. time step, 
# 2. solver and solver tolerance
# 3. implicit vs explicit
# 4. iteratively solve for U, gg, H, L

UggHL = np.concatenate((U,gg,H,[L]))

data = np.load('spinup.npz')
x = data['x']
X = data['X']
X_ = data['X_']
dx = data['dx']
Ut = data['Ut']
U = data['U']
H = data['H']
W = data['W']
L = data['L']
gg = data['gg']
B = data['B']
Ugg = np.concatenate((U,gg))


for k in np.arange(1,n+1):
    print('Time: ' + "{:.2f}".format(k*dt) + ' years')     
    
    #### START implicit time step
    result = root(model.implicit, UggHL, (x,X,Ut,H,W,dx,dt,U,H,L,B), method='lm', tol=1e-9)#, jac_options={'method':'gmres'})#, tol=1e-4, options={'maxiter':int(1e3)})
    U = result.x[:len(x)]
    gg = result.x[len(x):2*len(x)-1]
    H = result.x[2*len(x)-1:-1]
    L = result.x[-1]

    xt = X[0] + U[0]*dt
    xL = xt + L
    X = np.linspace(xt,xL,len(x))-xt
    X_ = (X[:-1]+X[1:])/2

    #### END implicit time step
    
    
    #### START explicit time step
    # result = root(model.diagnostic, Ugg, (x,X,Ut,H,W,dx,L), method='lm', tol=1e-6)
    # Ugg = result.x
    # U = result.x[:len(x)]
    # gg = result.x[len(x):]
    
    # advection_term = (U[0] + x_*(U[-1]-U[0]))*np.gradient(H,L*dx)
    # coordinate_stretching_term = ((U[2:]+U[1:-1])*H[1:]*W[1:]-(U[1:-1]+U[:-2])*H[:-1]*W[:-1])/(2*L*dx*W[1:])
    # coordinate_stretching_term = np.append(2*coordinate_stretching_term[0]-coordinate_stretching_term[1],coordinate_stretching_term) # a bit of a hack
    # dHdt = B + advection_term - coordinate_stretching_term
    # H += dHdt*dt
    
    # L = L+(U[-1]-U[0])*dt
    # X = x*L
    # X_ = (X[:-1]+X[1:])/2
    #### END explicit time step
    
    muW = model.get_muW(x,U,H,W,L,dx)
    
    W = width((X[:-1]+X[1:])/2)
    
    ind = int(len(W)/2) # index of midpoint in fjord
    y, u_transverse, _ = model.transverse(W[ind],muW[ind],H[ind])
    
    if (k % 2) == 0:
        ax1.plot(X,U/config.daysYear,color=plt.cm.viridis(color_id[k]))
        
        
        ax2.plot(np.append(X_,X_[::-1]),np.append(-config.rho/config.rho_w*H,(1-config.rho/config.rho_w)*H[::-1]),color=plt.cm.viridis(color_id[k]))
        
        ax3.plot(X_,gg,color=plt.cm.viridis(color_id[k]))
        ax4.plot(X[1:-1],muW,color=plt.cm.viridis(color_id[k]))
        ax5.plot(np.append(y,y+y[-1]),np.append(u_transverse,u_transverse[-1::-1])/config.daysYear,color=plt.cm.viridis(color_id[k]))

#ax3.plot(np.array([0,1e4]),np.array([config.muS,config.muS]),'k:')
ax4.plot(np.array([0,1e4]),np.array([config.muS,config.muS]),'k:')

#plt.savefig('test.png',format='png',dpi=150)    