# config.py contains global variables and other parameters


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:46:51 2022

@author: jason
"""


deps = 0.01     # finite strain rate parameter [a^-1]

d = 25 # characteristic iceberg size [m]
Hc = d # critical thickness [m]

# parameters for granular fluidity rheology
A = 0.5 
b = 1e5
muS = 0.1
muW_ = 2*muS # guess for muW iterations
muW_max = 1 # maximum value for muW # better convergence if no maximum
