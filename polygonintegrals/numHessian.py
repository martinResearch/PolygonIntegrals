
import numpy as np
import scipy

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def numHessian(f, x0, epsilon=1e-6):

    # Local Variables: f, DE0, H, k, DE1, epsilon, x, x0, E1, E0
    # Function calls: nargin, zeros, numel, Hessian
    #% evaluate the hessian of a function 
    H = np.zeros((x0.size, x0.size))
    E0, DE0 = f(x0)

    
    x0=np.array(x0,dtype=np.float64)
    for k in np.arange(0, x0.size):
        x = x0.copy()
        x.put(k,x.item(k)+epsilon) 
        
        E1, DE1 = f(x)
        H[:,k] = (DE1.flatten(0)-DE0.flatten(0))/ epsilon
        
    return H
