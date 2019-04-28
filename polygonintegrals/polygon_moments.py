
import numpy as np
import scipy


# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def nchoosek(n,k):
    return binomial(int(n),int(k))

def binomial(n,k): # from http://userpages.umbc.edu/~rcampbel/Computers/Python/probstat.html
    """Compute n factorial by a direct multiplicative method.""" 
    if k > n-k: k = n-k # Use symmetry of Pascal's triangle 
    accum = 1 
    for i in range(1,k+1): 
        accum *= (n - (k - i)) 
        accum /= i 
    return accum

def matixpower(m,p):
    if m.size==1:
        return m**p
    else:
        raise

def matdiv(a,b):
    return a/b


def polygon_moments(x, y, order, type='regular'):

    # Local Variables: Mu, j, Center, Area, i, s2, k, nbp, M, l, varargout, q, p, im, V, y, x, type, order, s1
    # Function calls: nchoosek, sum, nargin, zeros, numel, polygon_moments, strcmpi
    #%  POLYGON_MOMENTS Computes the moments of a polygon
    #% 
    #%  USES :
    #%
    #%    [V]=polygon_moments(x,y,order,'regular'(default))
    #%    [Mu,Center]=polygon_moments(x,y,order,'central')
    #%    [Mu,Area,Center]=polygon_moments(x,y,order,'central_normalized')
    #%
    #%  INPUT :
    #%
    #%    x and y : specify the coordinates of the polygon vertices
    #%    order   : specify up to which order we want the moements
    #%
    #%  OUTPUT :
    #%
    #%   V  : the moygon moments
    #%   Mu : the normalized central moments 
    #%   Area: Area of the polygon surface
    #%   Center: Centroid of the polygon
    #%
    #%   V(p+1,q+1)=integral of (x^p*y^q)  within the polygon
    #%   Mu(p+1,q+1)=(1/area)* integral of ((x-mean(x))^p*(y-mean(y)))^q) 
    #%   within the polygon
    #%
    #%   Note that  V(1,1) is the area of the polygon
    #%
    #%   This is a direct implementation of
    #%   "On the Calculation of Arbitrary Moments of Polygons" by Carsten Steger
    #%   Technical Report FGBV 96 05October 1996
    #%   Implemented by Martin de La Gorce 29 09 2009
    #%
    #%
    #%   EXAMPLE (from the Carsten Steger's paper):
    #%
    #%   x=[2,10,8,0];
    #%   y=[0,4,8,4];
    #%   plot([x,x(1)],[y,y(1)]);
    #%   [Mu,Area,Center]=polygon_moments(x,y,2,'central_normalized')
    #%   [V]=polygon_moments(x,y,2)
    #%   [M]=polygon_moments(x,y,2,'scale_invariant')
    #%   [M]=polygon_moments(x*2,y*2,2,'scale_invariant')
    #%

    ## WARNING : could use opencv's function called moments


    
    nbp = x.size
    #% compute moments
    if strcmpi(type,'regular'):
        M = np.zeros(((order+1.), (order+1.)))
        for p in np.arange(0, (order)+1):
            for q in np.arange(0, (order)+1):
                i = np.arange(1, (numel(x))+1)
                im = i-1
                im[im == 0] = nbp
                s2 = 0.
                for k in np.arange(0, (p)+1):
                    for l in np.arange(0, (q)+1):
                        s2 = s2+np.dot(np.dot(nchoosek((k+l), l), nchoosek((p+q-k-l), (q-l))), x[i-1]**k*x[im-1]**(p-k)*y[i-1]**l*y[im-1]**(q-l))
                        
                    
                s1 = np.sum(((x[im-1]*y[i-1]-x[i-1]*y[im-1])*s2))
                M[((p+1))-1,((q+1))-1] = matdiv(s1, np.dot(np.dot(p+q+2., p+q+1.), nchoosek((p+q), p)))
                
            
        return  M
    elif strcmpi(type,'central'):
        V = polygon_moments(x, y, 2., 'regular')
        Area = V[0,0]
        Center = matdiv(np.array(np.hstack((V[1,0], V[0,1]))), Area)
        M = polygon_moments((x-Center[0]), (y-Center[1]), order, 'regular')
        return M, Center
        
    elif strcmpi(type,'central_normalized'):
        V = polygon_moments(x, y, 2., 'regular')
        Area = V[0,0]
        Center = matdiv(np.array(np.hstack((V[1,0], V[0,1]))), Area)
        M = matdiv(polygon_moments((x-Center[0]), (y-Center[1]), order, 'regular'), Area)
        return M, Area, Center
        
    elif strcmpi(type,'scale_invariant'):
        V = polygon_moments(x, y, 2., 'regular')
        Area = V[0,0]
        Center = matdiv(np.array(np.hstack((V[1,0], V[0,1]))), Area)
        Mu = polygon_moments((x-Center[0]), (y-Center[1]), order, 'regular')
        M = np.zeros((order+1, order+1))
        for i in np.arange(0, (order)+1):
            for j in np.arange(0, (order)+1):
                M[((i+1.))-1,((j+1.))-1] = np.dot(Mu[((i+1))-1,((j+1))-1], matixpower(Area, -(1.+np.dot(0.5, i+j))))
                
            
        return M, Area, Center
        
    elif strcmpi(type,'Flusser'):
        #% to be implemented from 
        #% J. Flusser: "On the Independence of Rotation
        #% Moment Invariants"        
        pass
    
    

if __name__ == '__main__':

    # Local Variables: Center, Area, M, Mu, V, y, x
    # Function calls: test, plot, polygon_moments
    x = np.array(np.hstack((2., 10., 8., 0.)))
    y = np.array(np.hstack((0., 4., 8., 4.)))
    plt.plot(np.array(np.hstack((x, x[0]))), np.array(np.hstack((y, y[0]))))
    [Mu, Area, Center] = polygon_moments(x, y, 2., 'central_normalized')
    print(Mu, Area,Center)
    V = polygon_moments(x, y, 2.)
    M = polygon_moments(x, y, 2., 'scale_invariant')
    M = polygon_moments((x*2.), (y*2.), 2., 'scale_invariant')
    



