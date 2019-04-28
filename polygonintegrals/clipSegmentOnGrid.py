#cython: profile=True
#cython: boundscheck=False    
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True


import numpy as np
import scipy
from numpy import floor,ceil
#from numba import autojit

#cython#from libc.math cimport floor,ceil,abs
#cython#cimport numpy as np
#cython#cimport cython

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass
#@autojit
def clipSegmentOnGrid(\
    #cython#double\
    xa,\
    #cython#double\
    ya,\
    #cython#double\
    xb,\
    #cython#double\
    yb,
    #cython#int\
    nargout=1,display=False): 
    
    """Clip a line segment on the pixel grid. This will return a polyline whose 
    segments are not crossing any vertical line corresponding to x interger 
    and  any horizontal line with y integer 
    
    >>> xa = 1.5
    >>> ya = 0.5
    >>> xb = 3.5
    >>> yb = 4.5
    >>> L, t = clipSegmentOnGrid(xa, ya, xb, yb,nargout=2,display=False)  
    >>> print(L.T)
    [[ 1.5   1.75  2.    2.25  2.75  3.    3.25  3.5 ]
     [ 0.5   1.    1.5   2.    3.    3.5   4.    4.5 ]]
    >>> print(t)
    [ 0.     0.125  0.25   0.375  0.625  0.75   0.875  1.   ]
    """    
     
    #cython#cdef double xh,yv,Delta_x,Delta_y
    #cython#cdef int  dx,dy,xv,yh
    #cython#cdef int Nv,Nh
   
    if xa<xb:
        #% b is a the right of a
        dx = 1
        #% x should be incremented while going from a to b
        xv = int(floor(xa))+1
        #% the first vertical line intersected is at the right of a
    else:
        #% b is a the left of a
        dx = -1
        #% x should be decremented while going from a to b
        xv = int(ceil(xa))-1
        #% the first vertical line intersected is at the left of a
        
    
    if dx*xv < dx*xb:
        #% the segment is not vertical and the first intersection with the ray
        #% is in the segment [a,b]
        Delta_y = (yb-ya)/(xb-xa)
        yv = (xv-xa)* Delta_y+ya
        Nv = int(ceil(dx*( xb-xv)))
    else:
        Nv = 0
       
        
    
    #% Get the first intersection of the ray starting at a passing through b 
    #% with an horizontal line 
    if yb > ya:
        dy = 1
        yh = int(floor(ya))+1
    else:
        dy = -1
        yh = int(ceil(ya))-1
        
    
    if dy*yh < dy*yb:
        Delta_x = (xb-xa)/(yb-ya)
        xh = (yh-ya)* Delta_x+xa
        Nh = int(ceil(dy*( yb-yh)))
    else:
        xh = dx+xb
        Nh = 0
        
    #cython#cdef np.ndarray[np.float64_t, ndim=2]\
    L = np.empty(( Nh+Nv+2,2))
    #% loop until it reach the extremity b
    # L[0] = (xa, ya)
    L[0,0] = xa
    L[0,1] = ya

    #cython#cdef unsigned int k
    for k in xrange( 1,Nh+Nv+1):
        if dx* xv<dx* xh:
            #L[k] = (xv, yv)
            L[k,0] = xv # compiles in fewer lines in cython
            L[k,1] = yv # compiles in fewer lines in cython
            xv = xv+dx
            yv = yv+dx*Delta_y
        else:
            #% the next intersection is with an horizontal line
            #L[k] = (xh, yh)
            L[k,0] = xh # compiles in fewer lines in cython
            L[k,1] = yh # compiles in fewer lines in cython  
            yh = yh+dy
            xh = xh+dy*Delta_x           
                
    L[Nh+Nv+1,0] = xb
    L[Nh+Nv+1,1] = yb            
   
    #% could compute t in the loops above 
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    t = np.empty(( Nh+Nv+2), dtype=np.float64)        
    
    if (nargout==2) or display:
        if abs((xb-xa)) > abs((yb-ya)):
            t[:] = (L[:,0]-xa)/( xb-xa)
        else:
            t[:] = (L[:,1]-ya)/( yb-ya) 
            
    if display:
            plt.plot([xa, xb],[ya, yb], 'b-x')            
            plt.plot(L[:,0], L[:,1], '.')
            epsilon = (0.1/ np.linalg.norm(np.array(np.hstack((xb-xa, yb-ya)))))
            for k in xrange(len(t)-1):
                    plt.plot(xa+(xb-xa)*np.array((t[k]+epsilon, t[k+1]-epsilon)), ya+(yb-ya)*np.array((t[k]+epsilon, t[k+1]-epsilon)))
            plt.grid()   
            plt.show()    
                
    
    if nargout==1:
        return L   
    elif nargout==2:            
        return L, t

def _test():
    import doctest
    doctest.testmod()
    
if __name__ == '__main__':
    _test()

    


    
