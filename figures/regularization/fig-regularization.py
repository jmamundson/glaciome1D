#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 14:04:35 2023

@author: jason
"""

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
from glaciome1D import logistic, calc_df, parameters, deformational_thickness, constants


from scipy.integrate import quad
import pickle

matplotlib.rc('lines',linewidth=1) 

font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 8}

matplotlib.rc('font', **font)

constant = constants()
param = parameters()

file = open('steady-state_Bdot_-0.60.pickle','rb')
data = pickle.load(file)
file.close()

#%%
muS = param.muS
d = param.d
k = 50
mu = np.linspace(0,2,1001)

f_ = [quad(calc_df,1e-4,x,args=(param.muS,2*k))[0] for x in mu] 


f = 1-muS/mu 
f[mu<=muS] = 0

df = muS/mu**2
df[mu<muS] = 0
df[mu==muS] = np.nan

df_ = np.array([calc_df(x,muS,k) for x in mu])


#%%

cm = 1/2.54
fig_width = 19*cm
fig_height = 9*cm
plt.figure(figsize=(fig_width,fig_height))

left = 2*cm/fig_width
bottom = 1.5*cm/fig_height
width = 7*cm/fig_width
height = 7*cm/fig_height
xgap = 2*cm/fig_width

ax1 = plt.axes([left,bottom,width,height])
ax2 = plt.axes([left+width+xgap, bottom, width, height])


ax1.plot(mu,df,'k--')
ax1.plot(mu,df_,'k')
ax1.set_xlim([0,1])
ax1.set_xlabel('$\mu$')
ax1.set_ylim([-0.25,5.25])
ax1.set_ylabel('$df/d\mu$')
ax1.text(0.05,0.95,'a',transform=ax1.transAxes,va='top',ha='left')



ax2.plot(mu,f,'k--')
ax2.plot(mu,f_,'k')
ax2.set_xlim([0,1])
ax2.set_xlabel('$\mu$')
ax2.set_ylim([-0.04,0.84])
ax2.set_ylabel('$f(\mu)$')
ax2.text(0.05,0.95,'b',transform=ax2.transAxes,va='top',ha='left')


#plt.savefig('fig-gloc_regularization.pdf',format='pdf',dpi=300)

#%%
plt.close()

#%%
k = 100

H = np.linspace(0,100,101)
Hd = deformational_thickness(H,data)


fig_width = 19*cm
fig_height = 9*cm
plt.figure(figsize=(fig_width,fig_height))

left = 2*cm/fig_width
bottom = 1.5*cm/fig_height
width = 7*cm/fig_width
height = 7*cm/fig_height
xgap = 2*cm/fig_width

ax1 = plt.axes([left,bottom,width,height])
ax2 = plt.axes([left+width+xgap, bottom, width, height])

Hd_ = [np.max([x-d,0]) for x in H]
ax1.plot(H,Hd_,'k--')
ax1.plot(H,Hd,'k')
ax1.text(0.05,0.95,'a',transform=ax1.transAxes,va='top',ha='left')
ax1.set_xlim([0,100])
ax1.set_xlabel('Thickness [m]')
ax1.set_ylabel('H-d [m]')
ax1.set_ylim([-20,80])


ax2.plot(mu,np.abs(mu-muS),'k--')
ax2.plot(mu,2/k*np.logaddexp(0,k*(mu-muS)) - 2/k*np.logaddexp(0,-k*muS) + muS - mu,'k')
ax2.set_xlim([0.,0.5])
ax2.set_xlabel('$\mu$')
ax2.set_xticks(np.linspace(0,0.5,6,endpoint=True))
ax2.set_ylim([-0.0,0.5])
ax2.set_ylabel('$|\mu-\mu_s|$')
ax2.set_yticks(np.linspace(0,0.5,6,endpoint=True))
ax2.text(0.05,0.95,'b',transform=ax2.transAxes,va='top',ha='left')



plt.savefig('fig-regularization.pdf',format='pdf',dpi=300)
