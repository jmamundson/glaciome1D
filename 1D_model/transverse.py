# qsf: quasi-static flow

import numpy as np
from scipy import sparse
from scipy.optimize import fsolve

#%% define global variables
rho = 917.
rho_w = 1028.
g = 9.81

secs_in_day = 86400.
u_t = 50. # mean terminus speed [m/d]


#%% convert effective pressure to ice melange thickness
def thickness(W,L,H_L,mu_0):
    #H = 2*P/(rho*(1-rho/rho_w)*g)
    
    H_x = lambda H,x : H - H_L*np.exp(mu_0*(L-x)/W+1/2.-H_L/(2*H))
    
    H_guess = 200

    x = np.linspace(0,L,101)
    H = np.zeros(len(x))
    
    for j in range(0,len(x)):
        H[j] = fsolve(H_x,H_guess,x[j])
    
    return(x,H)

#%% calculate the effective pressure driving flow

# W = ice melange width
# L = ice melange length
# mu_0 = coefficient of friction at yield stress
def pressure(x,H):
    
    P = 0.5*rho*g*(1-rho/rho_w)*H
    
    return(P)

#%% calculate resistive force (a.k.a. buttressing force) against the glacier terminus

# ratio = L/W
def force(H_L,mu_0):
    
    H_0 = lambda H,ratio : H - H_L*np.exp(mu_0*ratio+1/2.-H_L/(2*H))
    
    ratio = np.linspace(0,8,501)
    H_ratio = np.zeros(len(ratio))
    
    H_guess = 200
    
    for j in range(0,len(ratio)):
        H_ratio[j] = fsolve(H_0,H_guess,ratio[j])
        
        
    F = 0.5*rho*g*(1-rho/rho_w)*H_ratio*(H_ratio-H_L)
    
    return(ratio,F)



#%% solve velocity profile for mu(I) rheology

# NEED TO BE CAREFUL WITH GLOBAL VS LOCAL VARIABLES!!!
# create dictionary to pass variables?

# mu_s = minimum coefficient of friction
# mu_2 = essentially just the maximum coefficient of friction, but
#     it should be slightly smaller than mu_0
# mu_0 = maximum coefficient of friction
# d = grain size
# I0 = dimensionless parameter
# mu_w = coefficient of friction along the fjord wall


def muI(H, W, mu_s, mu_w, mu_0, d, I0, **kwargs):
    #H,W,mu_s,mu_w,mu_0,d,I0
    
    P = 0.5*rho*g*(1-rho/rho_w)*H   
    
    y_c = W/2*(1-mu_s/mu_w)
    
    n_pts = 101 # number of points in half-space
    y = np.linspace(0,y_c,n_pts) # location of points
    
    dy = y[1]
    
    Gamma = -2*I0*np.sqrt(P/rho)/d # leading term in equation of velocity profile
    
    mu = mu_w*(1-2*y/W)

    # RHS of differential equation
    b = Gamma*(mu-mu_s)/(mu-mu_0) # should mu_0 be mu_2? 
    b[0] = 0 # set boundary condition of u=0; later this will be adjusted to ensure mass continuity
    b[-1] = 0 # set boundary condition of du/dy=0
    b = b*2*dy
    
    # centered differences
    a = np.zeros(len(y))
    a[0] = 1
    a[-1] = 1
   
    a_left = -np.ones(len(y)-1)
    
    a_right = np.ones(len(y)-1)
    a_right[0] = 0
    
    diagonals = [a_left,a,a_right]
    A = sparse.diags(diagonals,[-1,0,1]).toarray()
    
    u = np.linalg.solve(A,b)
    u = u*secs_in_day
    
    u = np.append(u,u[-1])
    y = np.append(y,W/2)
    
    u_mean = np.mean(u)*2*y_c/W + np.max(u)*(1-2*y_c/W)
    
    return(y,u,u_mean)
    
def muI_minimize(mu_w):
    
    muI_variables = {'H':H, 'W':W, 'mu_s':mu_s, 'mu_w':mu_w, 'mu_0':mu_0, 'd':d, 'I0':I0}
    #_, _, u_mean = muI(**muI_variables)
    _, _, u_mean = muI(H, W, mu_s, mu_w, mu_0, d, I0)
    du = np.abs(U-u_mean)    
    return(du)

#%% solve transverse velocity for nonlocal rheology

# A and b are dimensionless parameters
def nonlocal_(W,mu_s,mu_0,P,d,A,b): 
    
    n_pts = 101 # number of points in half-space
    y = np.linspace(0,W/2,n_pts) # location of points
    
    dy = y[1]

    mu = mu_0*(1-2*y/W)

    y_c = W/2*(1-mu_s/mu_0) # critical value of y for which mu is no longer greater 
    # than mu_s; although flow occurs below this critical value, it is needed for 
    # computing g_loc (below)
    
    zeta = np.sqrt(np.abs(mu-mu_s))/(A*d)
    
    g_loc = np.zeros(len(y))
    g_loc[y<y_c] = np.sqrt(P/rho)*(mu[y<y_c]-mu_s)/(mu[y<y_c]*b*d) # local granular fluidity
    
    # first solve for the granular fluidity. we set dg/dy = 0 at
    # y = 0 and at y = W/2    
    
    a = -(2+dy**2*zeta**2)
    a[0] = -1
    a[-1] = 1
    
    a_left = np.ones(len(y)-1)
    a_left[-1] = -1
    
    a_right = np.ones(len(y)-1)
    
    diagonals = [a_left,a,a_right]
    A = sparse.diags(diagonals,[-1,0,1]).toarray()
    
    f = -g_loc*zeta**2*dy**2
    f[0] = 0
    f[-1] = 0
    
    gg = np.linalg.solve(A,f) # solve for granular fluidity
    
    a = np.zeros(len(y))
    a[0] = 1
    a[-1] = 1
    
    a_left = -np.ones(len(y)-1)
    
    a_right = np.ones(len(y)-1)
    a_right[0] = 0
    
    diagonals = [a_left,a,a_right]
    A = sparse.diags(diagonals,[-1,0,1]).toarray()
    
    f = mu*gg*2*dy
    f[0] = 0
    f[-1] = 0
    
    # boundary conditions: u(0) = 0; du/dy = 0 at y = W/2
    
    u = np.linalg.solve(A,f)
    u = u*secs_in_day
    
    u0 = u_t - np.mean(u)
    u = u + u0 
    
    return(y,u)


#%%
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint

H = 100
#P = 0.5*rho*g*(1-rho/rho_w)*H
mu_s = 0.2
mu_w = 0.5
mu_0 = 0.6
d = 25
I0 = 1e-6
W = 4000
U = 10 # mean velocity, in m/d

mu_w = 0.4
#bounds = Bounds(mu_s,mu_0*0.999)
linear_constraint = LinearConstraint([1], mu_s, mu_0*0.999)

result = minimize(muI_minimize, mu_w, method='COBYLA', constraints=[linear_constraint], tol=1e-6, options={'disp': True})
mu_w  = result.x
print(mu_w)
#%%
y, u, u_mean = muI(P,W,mu_s,mu_w,mu_0,d,I0)

y = y-y[-1]
y = np.append(y,np.abs(y[::-1]))
u = np.append(u,u[::-1])
plt.plot(y,u)

#%%
def arg_printer(**kwargs):
   print(kwargs)
   
dct = {'param1':5, 'param2':8}
arg_printer(**dct)
{'param1': 5, 'param2': 8}