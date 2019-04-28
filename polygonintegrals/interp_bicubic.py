
import numpy as np
import scipy


# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def interp_bicubic(f, x, y,nargout=1):

    # Local Variables: uy, ux, f, i, j, d2fb, dux, duy, fb, y, x, dfb
    # Function calls: interp_bicubic, floor, d2u, nargout, zeros, u, du
    #% work as interp2(f,y,x,'bicubic') but also provides the derivative at (x,y)
    #%
    #% coded by Martin de la Gorce
    #%
    #% x=-3:0.01:3;plot(x,arrayfun(@d2u,x));
    fb = 0.
    if nargout == 1.:
        for i in np.arange(np.floor(x)-1., (np.floor(x)+2.)+1):
            for j in np.arange(np.floor(y)-1., (np.floor(y)+2.)+1):
                fb = fb+np.dot(np.dot(f[int(i)-1,int(j)-1], u((x-i))), u((y-j)))
                
            
    elif nargout == 2.:
        fb = 0.
        dfb = np.array(np.hstack((0., 0.)))
        for i in np.arange(np.floor(x)-1., (np.floor(x)+2.)+1):
            for j in np.arange(np.floor(y)-1., (np.floor(y)+2.)+1):
                ux = u((x-i))
                uy = u((y-j))
                fb = fb+np.dot(np.dot(f[int(i)-1,int(j)-1], ux), uy)
                dfb[0] = dfb[0]+np.dot(np.dot(f[int(i)-1,int(j)-1], du((x-i))), uy)
                dfb[1] = dfb[1]+np.dot(np.dot(f[int(i)-1,int(j)-1], ux), du((y-j)))
                
            
        
    elif nargout == 3.:
        fb = 0.
        dfb = np.array(np.hstack((0., 0.)))
        d2fb = np.zeros((2., 2.))
        for i in np.arange(np.floor(x)-1., (np.floor(x)+2.)+1):
            for j in np.arange(np.floor(y)-1., (np.floor(y)+2.)+1):
                ux = u((x-i))
                uy = u((y-j))
                dux = du((x-i))
                duy = du((y-j))
                fb = fb+np.dot(np.dot(f[int(i)-1,int(j)-1], ux), uy)
                dfb[0] = dfb[0]+np.dot(np.dot(f[int(i)-1,int(j)-1], dux), uy)
                dfb[1] = dfb[1]+np.dot(np.dot(f[int(i)-1,int(j)-1], ux), duy)
                d2fb[0,0] = d2fb[0,0]+np.dot(np.dot(f[int(i)-1,int(j)-1], d2u((x-i))), uy)
                d2fb[0,1] = d2fb[0,1]+np.dot(np.dot(f[int(i)-1,int(j)-1], dux), duy)
                d2fb[1,0] = d2fb[1,0]+np.dot(np.dot(f[int(i)-1,int(j)-1], dux), duy)
                d2fb[1,1] = d2fb[1,1]+np.dot(np.dot(f[int(i)-1,int(j)-1], ux), d2u((y-j)))
                
            
        
    
    #% thuis corresponds the the convilution formula on 
    #% wikipedia with a=-0.5
    if nargout==1:
        return fb
    elif nargout==2:
        return fb,dfb
    elif nargout==3:
        return fb, dfb, d2fb
def u(s):

    # Local Variables: y, s, as
    # Function calls: abs, u
    _as = np.abs(s)
    if _as<=1.:
        y = np.dot(3./2., _as**3.)-np.dot(5./2., _as**2.)+1.
    elif _as<=2.:
        y = np.dot(-1./2., _as**3.)+np.dot(5./2., _as**2.)-4.*_as+2.
        
    else:
        y = 0.
        
    
    return [y]
def du(s):

    # Local Variables: s, as, dy
    # Function calls: du, abs
    _as = np.abs(s)
    if _as<=1.:
        dy = np.dot(9./2., _as**2.)-5.*_as
    elif _as<=2.:
        dy = np.dot(-3./2., _as**2.)+5.*_as-4.
        
    else:
        dy = 0.
        
    
    if s<0.:
        dy = -dy
    
        
    return [dy]

def d2u(s):

    # Local Variables: s, as, dy
    # Function calls: du, abs
    _as = np.abs(s)
    if _as<=1.:
        d2y = 9.*_as-5
    elif _as<=2.:
        d2y =-3*_as+5
        
    else:
        d2y = 0.
    
    return d2y
