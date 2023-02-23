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

dee = 1e-16      # finite strain rate parameter
dgg = 1e-16      # finite granular fluidity parameter

d = 25 # characteristic iceberg size [m]


# parameters for granular fluidity rheology
A = 0.5
b = 2e5
muS = 0.1
muW_ = 2*muS

    # constants --> Needs some thought!
    #b = (mu0-muS)/I0
    #A = 0.5
