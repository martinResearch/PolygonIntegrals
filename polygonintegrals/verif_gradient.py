
import numpy as np
import scipy


# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def verif_gradient(f, x0, range=[],verbose=True,epsilon=1e-6,angle_threshold=0.001):

    # Local Variables: c, dE, f, epsilon, k, DE, epsilon2, range, dE2, DE1, x, x0, E2, E1, E0, abs_angle
    # Function calls: verif_gradient, atan2, setappdata, nargin, abs, fprintf, numel, error, pi, mod
    if range==[]:
        range = np.arange(1., (len(x0))+1)
    
    valid=True
    E0, DE = f(x0)
    DE=DE.flatten()
    if len(DE) != (x0.size):
        raise 'the gradient as not the right size'
    
    if verbose:
        print('\nE0=%e \n'% E0)
   
    dE=np.zeros((x0.size))
    dE2=np.zeros((x0.size))
    for k in range:
        x = x0.copy()
        x.put(int(k-1),x.item(int(k)-1)+epsilon)
        E1, DE1 = f(x)
        dE[int(k)-1] = E1-E0
        #%compare(DE,DE1)
        #% k
        c = np.array(np.hstack((dE[int(k)-1]/ epsilon, DE[int(k)-1])))
        if verbose:
            print('\n\nDirection %d :\n'% k)
        abs_angle = np.abs((np.mod(np.arctan2(c[0],c[1]), np.pi)-np.pi/4.))
        if np.logical_and(c[0] == 0., c[1] == 0.):
            abs_angle = 0.
        
        
        if abs_angle > angle_threshold:
            epsilon2 = -10*epsilon          
            x = x0.copy()
            x.put(int(k-1),x.item(int(k)-1)+epsilon2)
            E2,DE2 = f(x)
            dE2[int(k)-1] = E2-E0
            if verbose:
                print(' divided difference : %e (eps=%2.3e)\n divided difference : %e (eps=%2.3e)\n \n provided gradient  :% e\n'% (dE[k-1]/ epsilon, epsilon, dE2[k-1]/ epsilon2, epsilon2, DE[k-1]))
                print(' angle %e :\n', abs_angle)
                print(' E1 %e E2 %e :\n'%( E1, E2))
                print(' la difference est grande!\n')
            valid=False
        else:
            if verbose:
                print(' divided difference : %e (eps=%2.3e)\n provided gradient  : %e \n'%(dE[int(k)-1]/ epsilon, epsilon, DE[int(k)-1]))
                print(' angle %e :\n'%abs_angle)
                print(' OK\n')
            
        
        
    return  valid
