import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import quad

#%%
def logistic(x,xc,k):
    '''
    Returns a logistic function. Used for regularization of g_loc and abs(mu-muS)
    
    Parameters
    ----------
    x : dimensionless grid
    xc : location where step change should occur
    k : amount of smoothing
    
    Returns
    -------
    h : logistic function
    '''
    
    h = 1/(1+np.exp(-k*(x-xc)))
    return(h)



def dP(H,d,k):
    
    f = 1/((1+np.exp(-k*(H-d))))
    
    return(f)

#%%
d = 25 # grain size
xc = 0.9
k = 100

H = np.linspace(0,100,1001)

#f = [quad(dP,1e-4,x,args=(d,k))[0] for x in H] 

#plt.plot(H,f)

f_ = H-d
f_[f_<0] = 0

plt.plot(H,np.log(1+np.exp(k*(H-d)))/k)
plt.plot(H,f_)
#plt.plot(H,logistic(H,d,0.9))

#plt.ylabel('$\gamma$')
#plt.xlim([0,100])