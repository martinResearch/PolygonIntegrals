
import numpy as np
import scipy


# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def interp_bilinear(f, x, y,nargout=1):

    # Local Variables: f, fx, fy, fb, f01, f00, y, x, f10, f11, dfb
    # Function calls: interp_bilinear, nargout, floor
    #% work as interp2(f,y,x,'bilinear') but also provides the derivative at (x,y)
    #%
    #% coded by Martin de la Gorce
    #%
    fx = np.floor(x)
    fy = np.floor(y)
    f00 = f[int(fx)-1,int(fy)-1]
    f11 = f[int((fx+1.))-1,int((fy+1.))-1]
    f10 = f[int((fx+1.))-1,int(fy)-1]
    f01 = f[int(fx)-1,int((fy+1.))-1]
    fb = np.dot(np.dot(f11, x-fx), y-fy)+np.dot(np.dot(f01, 1.-x+fx), y-fy)+np.dot(np.dot(f00, 1.-x+fx), 1.-y+fy)+np.dot(np.dot(f10, x-fx), 1.-y+fy)
    if nargout > 1.:
        dfb = np.array(np.hstack((0., 0.)))
        dfb[0] = np.dot(f11-f01, y-fy)+np.dot(f10-f00, 1.-y+fy)
        dfb[1] = np.dot(f11-f10, x-fx)+np.dot(f01-f00, 1.-x+fx)
    
    if nargout==1:
        return fb
    else:    
        return fb, dfb
