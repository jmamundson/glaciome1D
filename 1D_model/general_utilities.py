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

from config import *

from matplotlib import pyplot as plt

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

#%% calculate the effective pressure driving flow

# W = ice melange width
# L = ice melange length
# mu0 = coefficient of friction at yield stress
def pressure(H):
    
    P = 0.5*rho*g*(1-rho/rho_w)*H
    
    return(P)

#%% calculate resistive force (a.k.a. buttressing force) against the glacier terminus

# ratio = L/W
def force(H_L,mu0):
    
    H_0 = lambda H,ratio : H - H_L*np.exp(mu0*ratio+1/2.-H_L/(2*H))
    
    ratio = np.linspace(0,8,501)
    H_ratio = np.zeros(len(ratio))
    
    H_guess = 200
    
    for j in range(0,len(ratio)):
        H_ratio[j] = fsolve(H_0,H_guess,ratio[j])
        
        
    F = 0.5*rho*g*(1-rho/rho_w)*H_ratio*(H_ratio-H_L)
    
    return(ratio,F)