#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 11:58:53 2022

@author: jason
"""

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve
from scipy.sparse import diags

import config

from matplotlib import pyplot as plt




#%% calculate second invariant of the strain rate with respect to chi (stretched grid)
def second_invariant(U,dx):
    ee_chi = np.sqrt((np.diff(U)/dx)**2/2)+config.dee
    
    #ee_chi = np.sqrt(np.gradient(U,dx)**2/2) + config.dee
    #ee_chi = ee_chi[:-1]
    
    return(ee_chi)


#%% calculate the effective pressure driving flow

# W = ice melange width
# L = ice melange length
# mu0 = coefficient of friction at yield stress
def pressure(H):
    
    P = 0.5*config.rho*config.g*(1-config.rho/config.rho_w)*H
    
    return(P)



#%% Quasi-static tools 
# calculate resistive force (a.k.a. buttressing force) against the glacier terminus

# ratio = L/W
def force(H_L,mu0):
    
    H_0 = lambda H,ratio : H - H_L*np.exp(mu0*ratio+1/2.-H_L/(2*H))
    
    ratio = np.linspace(0,8,501)
    H_ratio = np.zeros(len(ratio))
    
    H_guess = 200
    
    for j in range(0,len(ratio)):
        H_ratio[j] = fsolve(H_0,H_guess,ratio[j])
        
        
    F = 0.5*config.rho*config.g*(1-config.rho/config.rho_w)*H_ratio*(H_ratio-H_L)
    
    return(ratio,F)

#%% convert effective pressure to ice melange thickness in quasi-static case
def thickness(W,L,H_L,mu0):
    #H = 2*P/(rho*(1-rho/rho_w)*g)
    
    H_x = lambda H,x : H - H_L*np.exp(mu0*(L-x)/W+1/2.-H_L/(2*H))
    
    H_guess = 200

    x = np.linspace(0,L,101)
    H = np.zeros(len(x))
    
    for j in range(0,len(x)):
        H[j] = fsolve(H_x,H_guess,x[j])
    
    return(x,H)

#%% ice melange calving

#%% width

def width(X):
    
    W = 4000*np.ones(len(X))
    
    #W = -4/8*X+8000
    #W[X<0] = 8000
    #W[X>8000] = 2000
    
    
    
    return(W)
    


