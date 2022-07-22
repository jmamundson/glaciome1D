# config.py contains global variables and other parameters


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:46:51 2022

@author: jason
"""
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize


rho = 917.       # density of ice
rho_w = 1028.    # density of water
g = 9.81         # gravitational acceleration

secsDay = 86400.
daysYear = 365.25
secsYear = secsDay*daysYear


d = 25 # characteristic iceberg size [m]

# parameters for muI rheology
mu0 = 0.6 # maximum coefficient of friction
muS = 0.4 # minimum coefficient of friction
muW_ = 0.4 # initial guess for coefficient of friction along the fjord walls
muI_constraint = LinearConstraint([1], muS, mu0*0.9999) # muS <= muW < mu0
I0 = 10**-6


# parameters for granular fluidity rheology
#muS = 0.2
A = 1
b = 2e5 
nonlocal_constraint = LinearConstraint([1], muS, muS*100)