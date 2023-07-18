# config.py contains global variables and other parameters


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:46:51 2022

@author: jason
"""


rho = 917.       # density of ice
rho_w = 1028.    # density of water
g = 9.81         # gravitational acceleration

secsDay = 86400.
daysYear = 365.25
secsYear = secsDay*daysYear

dgg = 1e-16      # finite granular fluidity parameter

d = 25 # characteristic iceberg size [m]
Hc = d # critical thickness [m]

# parameters for granular fluidity rheology
A = 0.5
b = 2e5
muS = 0.05
muW_ = 2*muS # guess for muW iterations
muW_max = 1 # maximum value for muW # better convergence if no maximum
