#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 10:31:41 2023

@author: jason
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib

import sys

sys.path.append('/home/jason/projects/glaciome/glaciome1D')
from glaciome1D import deformational_thickness

import config

matplotlib.rc('lines',linewidth=1) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

H = np.linspace(0,100,101)

Hd = deformational_thickness(H)

cm = 1/2.54
fig_width = 10*cm
fig_height = 9*cm
plt.figure(figsize=(fig_width,fig_height))

left = 2*cm/fig_width
bottom = 1.5*cm/fig_height
width = 7*cm/fig_width
height = 7*cm/fig_height
ax = plt.axes([left,bottom,width,height])

ax.plot(H/config.d,Hd/config.d,color='k')
ax.plot(H/config.d,(H/config.d-1),':',color='k')
ax.set_xlim([0,4])
ax.set_xlabel('$H/d$')
ax.set_ylim([-1,3])
ax.set_ylabel(r'$\frac{d}{k}\log\left( 1+e^{k(H/d-1)} \right)$')
ax.set_yticks([-1,0,1,2,3])
ax.set_xticks([0,1,2,3,4])

#plt.savefig('fig-Pd.pdf',format='pdf',dpi=300)
