#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 13:55:22 2023

@author: jason
"""

import numpy as np
from matplotlib import pyplot as plt

mu = np.linspace(0,1,101)
muS = 0.3

f = 2/(1+np.exp(-100*(mu-muS)))-1

f = 2/100*np.log(1+np.exp(100*(mu-muS)))-mu+muS-2/100*np.log(1+np.exp(-100*muS))

plt.plot(mu,f)