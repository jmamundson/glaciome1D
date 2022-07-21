#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 15:40:56 2022

@author: jason
"""


def granular_fluidity(x,U,H,dx):
    
    # note: use strain rate to compute mu, which is used to compute g'
    # need to iterate because sign(exx) is not necessarily always the same
    
    # constants
    b = (mu0-muS)/I0
    A = 0.5

    ee = np.abs(np.gradient(U,dx))
    mu = 0.5*np.ones(len(x)) # initial guess for mu
    gg = ee/mu
    
    k = 1
    while k==1: 
        
        g_loc = np.zeros(len(x))
        g_loc[mu>muS] = np.sqrt(0.5*g*(1-rho/rho_w)*H[mu>muS]/d**2)*(mu[mu>muS]-muS)/(mu[mu>muS]*b) 
        
        zeta = np.abs(mu-muS)/(A**2*d**2) # zeta = 1/xi^2
    
        # construct equation Cx=T
        # boundary conditions: 
        #    # g=0 at x=0, L (implies strain rate = 0)
        #    dg/dx=0 at x=0,L (implies strain rate gradient equals 0)
        
        c_left = np.ones(len(x)-1)
        c_left[-1] = -1
        
        c = -(2+zeta*dx**2)
        c[0] = -1
        c[-1] = 1
            
        c_right = np.ones(len(x)-1)
            
        diagonals = [c_left,c,c_right]
        C = diags(diagonals,[-1,0,1]).toarray() 
        
        T = -g_loc*zeta*dx**2
        T[0] = 0
        T[-1] = 0
            
        gg_new = np.linalg.solve(C,T) # solve for granular fluidity
     
        print(np.max(np.abs(gg-gg_new)))
        print(k)
        #if (np.abs(gg-gg_new) > 1e-10).any():        
        if np.max(np.abs(gg-gg_new)) > 1e-10:
            print(np.max(np.abs(gg-gg_new)))
            gg = gg_new
            gg[gg==0] = 1e-10 # small factor to make mu real; doesn't do anything because gg=0 when ee=0, so mu=0 in this case
            mu = ee/gg
            
        else:
            gg = gg_new
            gg[gg==0] = 1e-10
            mu = ee/gg
            xn = np.linspace(x[0]+dx/2, x[-1]+dx/2, len(x)-1 )
            nu = np.interp(xn,x,H**2/gg) # create new variable on staggered grid to simplify later
            
            break
        
        return(nu, mu)