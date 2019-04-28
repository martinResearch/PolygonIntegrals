#cython: profile=True
#cython: boundscheck=True    
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True

import numpy as np

import scipy.sparse as sp
import scipy.optimize

from clipSegmentOnGrid import *
from verif_gradient import *
from numHessian import *

#from numba import jit,autojit
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

#cython#from libc.math cimport floor,ceil,abs
#cython#cimport numpy as np
#cython#cimport cython

def error(mesg):
    raise  mesg

def sparse(m,n):
    return  sp.lil_matrix((m,n),dtype=np.float)

class ImageWithIntegral():
    def __init__(self,f):
        self.values=f
        self._cumsum=[]

    @property   
    def cumsum(self):#  method that allow leazy evaluation of its cumulative sum
        if self._cumsum==[]:
            self._cumsum= np.cumsum(self.values, 0)
        return self._cumsum
        
    
def integreWithinPolygon(\
    Im,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    P,  interp='bilinear',nargout=1):
    """
    Computes the integral of the image Im within the polygon P using the specified interpolation method  
    assuming Im is interpolated using f(x,y)=f(floor(x),floor(y))    
    
    INPUT :
    P : vertices of the polygon , matrix of size 2 by N
    f : the discretized 2D function beeing integrated
    F : can provide F=cumsum(f,1) to avoid recomputing F
    OUTPUT
    I     : the integral
    dI_dP : the derivative dI/dP
    M : the antialiased mask of the polygon, should get I==sum(M(:).*f(:))
    
    coded by Martin de la Gorce    
    
    >>> Im= ImageWithIntegral(np.ones((7,9),dtype=np.float64))
    >>> P=np.array([[2.,2.],[6.,2.],[6.,8.],[2.,8.]])
    >>> Ic, dIc_dP,trash,M = integreWithinPolygon(Im,P, 'const',nargout=4)
    >>> print(Ic)
    24.0
    >>> print(dIc_dP)
    [[-3. -2.]
     [ 3. -2.]
     [ 3.  2.]
     [-3.  2.]]
    >>> print(M)
    [[ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]
     [ 0.  0.  1.  1.  1.  1.  1.  1.  0.]
     [ 0.  0.  1.  1.  1.  1.  1.  1.  0.]
     [ 0.  0.  1.  1.  1.  1.  1.  1.  0.]
     [ 0.  0.  1.  1.  1.  1.  1.  1.  0.]
     [ 0.  0.  0.  0.  0.  0.  0.  0.  0.]]
    >>> Ib, dIb_dP, Hessianb = integreWithinPolygon(Im,P,  'bilinear',nargout=3)
    >>> print(Ib)
    24.0
    >>> print(dIb_dP)
    [[-3. -2.]
     [ 3. -2.]
     [ 3.  2.]
     [-3.  2.]]
     
    >>> tmp=np.array([\
     [ 0.,   0.,   0.,   0.5,  0.,   0.,   0.,  -0.5],\
     [ 0.,   0.,  -0.5,  0.,   0.,   0.,   0.5,  0. ],\
     [ 0.,  -0.5,  0.,   0.,   0.,   0.5,  0.,   0. ],\
     [ 0.5,  0.,   0.,   0.,  -0.5,  0.,   0.,   0. ],\
     [ 0.,   0.,   0.,  -0.5,  0.,  -0.,   0.,   0.5],\
     [ 0.,   0.,   0.5,  0.,   0.,   0.,  -0.5,  0. ],\
     [ 0.,   0.5,  0.,   0.,   0.,  -0.5,  0.,   0. ],\
     [-0.5,  0.,   0.,   0.,   0.5,  0.,   0.,   0. ]])   
    >>> np.testing.assert_almost_equal(Hessianb.todense(),tmp)
    
    >>> x = np.arange( 9.)
    >>> y = np.arange( 7.)
    >>> xv, yv = np.meshgrid(x, y)    
    >>> Ib, dIb_dP, Hessianb,M = integreWithinPolygon(ImageWithIntegral(xv),P,  'bilinear',nargout=4)
    >>> print(Ib)
    96.0
    >>> print(dIb_dP)
    [[ -9.  -2.]
     [  9.  -2.]
     [ 15.  14.]
     [-15.  14.]]
     
    >>> np.set_printoptions(precision=4,suppress=True,edgeitems=5)
    >>> print(Hessianb.todense())
    [[ 0.     -1.      0.      0.5     0.      0.      0.     -2.5   ]
     [-1.     -1.3333 -0.5    -0.6667  0.      0.      1.5     0.    ]
     [ 0.     -0.5     0.      1.      0.      2.5     0.      0.    ]
     [ 0.5    -0.6667  1.     -1.3333 -1.5     0.      0.      0.    ]
     [ 0.      0.      0.     -1.5     0.      1.      0.      3.5   ]
     [ 0.      0.      2.5     0.      1.      1.3333 -3.5     0.6667]
     [ 0.      1.5     0.      0.      0.     -3.5     0.     -1.    ]
     [-2.5     0.      0.      0.      3.5     0.6667 -1.      1.3333]]
     >>> print(M)
     [[ 0.    0.    0.    0.    0.    0.    0.    0.    0.  ]
      [ 0.    0.25  0.5   0.5   0.5   0.5   0.5   0.25  0.  ]
      [ 0.    0.5   1.    1.    1.    1.    1.    0.5   0.  ]
      [ 0.    0.5   1.    1.    1.    1.    1.    0.5   0.  ]
      [ 0.    0.5   1.    1.    1.    1.    1.    0.5   0.  ]
      [ 0.    0.25  0.5   0.5   0.5   0.5   0.5   0.25  0.  ]
      [ 0.    0.    0.    0.    0.    0.    0.    0.    0.  ]]
      
    >>> Ib, dIb_dP, Hessianb,M = integreWithinPolygon(ImageWithIntegral(yv),P, 'bilinear',nargout=4)
    >>> print(Ib)
    72.0
    >>> print(dIb_dP)
    [[ -3.      -4.6667]
     [ 15.      -7.3333]
     [ 15.       7.3333]
     [ -3.       4.6667]]
    """
          
    try:
        F = Im.cumsum
    except:
        raise 'your image should have a cumsum property, you can use the ImageWithIntegral class provided with this module'
        
    
    if P.shape[1] != 2 or P.shape[0]<3.:
        error('P should be of size N by 2 with N>=3')
    
    assert(np.max(P[:,0])<=F.shape[0]-1)
    assert(np.max(P[:,1])<=F.shape[1]-1)
    assert(np.min(P[:,0])>=1)
    
    N = P.shape[0]
    I = 0.
    ##cython#cdef np.ndarray[np.float64_t, ndim=2]\
    dI_dP = np.zeros((N, 2),dtype=np.float64)
    
    if interp.lower()== 'const':
        order=0
    elif interp.lower()== 'bilinear':
        order=1
    else:
        error('unkown interpolation method')            
        
    if nargout >= 3.:
        if order==0:
            # hessian not yet coded for const , cannot use the equation derived in the thesis manuscrip because fc is not diferentiable
           
            Hessian=[]
        elif order==1:
            HessianI_i_i = np.zeros((N,2,2)) 
            HessianI_ip_i = np.zeros((N,2,2)) 
            #Hessian2B=sp.lil_matrix((2*N,2*N))
    if nargout == 4.:
        S = np.zeros((F.shape[0], F.shape[1]) )   
    else: 
        S = np.zeros((0,0) ) 


   
        
    for i in range(N):
        ip = np.mod(i+1, N)
        if nargout == 1:
            L = clipSegmentOnGrid(P[i,0], P[i,1], P[ip,0], P[ip,1],1,False)
        else:
            L ,t= clipSegmentOnGrid(P[i,0], P[i,1], P[ip,0], P[ip,1],2,False)  
        L=L.T
        

        Is = _integreLeftLineFragements(L, F, order,S)
        I = I+Is       

        
        if nargout >= 2:
            if order==0:
           
                _computeGradientConst(F,P,L,t,i,ip,dI_dP)
            elif order==1:
                f00, f01, f10, f11, xc, yc, tc, Dx, Dy, Dt = _prepare_integral(Im.values, L, t) 
                C0, Cx, Cy, Cxy = _integrePolynomOnLineFragments3(xc, yc, tc, Dx, Dy, Dt, 1)
                _computeGradientBilinear(P,f00,f01,f10,f11,C0,Cx,Cy,Cxy,i,ip,dI_dP)                    
        if nargout >= 3: 
            if order==1:                      
               
                    DeltaP = P[ip]-P[i]
                    ay = f01-f00
                    axy = f11-f10+f00-f01
                    ax = f10-f00
                    a0 = f00
                    C0, Cx, Cy, Cxy= _integrePolynomOnLineFragments3(xc, yc, tc, Dx, Dy, Dt, 2)
                    p = np.array((0., 0., 1.))
                    axyCy=np.dot(axy, Cy)
                    axyCx=np.dot(axy, Cx)
                    axC0=np.dot(ax, C0)
                    ayC0=np.dot(ay, C0)
                    int_dfdx_t2 = np.dot(axyCy, p)+np.dot(axC0, p)# int_dfdx_t2[k]= \int_{t\in[t(k),t(k+1)]}\frac{df}{dx}t^2
                    int_dfdy_t2 = np.dot( axyCx, p)+np.dot(ayC0, p)
                    p = np.array((0., 1., -1.))
                    int_dfdx_t_1_m_t = np.dot(axyCy, p)+np.dot(axC0, p)
                    int_dfdy_t_1_m_t = np.dot( axyCx, p)+np.dot(ayC0, p)
                    p = np.array((1., -2., 1.))
                    int_dfdx_1_m_t_2 = np.dot(axyCy, p)+np.dot(axC0, p)
                    int_dfdy_1_m_t_2 = np.dot(axyCx, p)+np.dot(ayC0, p)
                    p = np.array((0., 1., 0.))
                    int_f_t = np.dot(np.dot(a0, C0), p)+np.dot(np.dot(ax, Cx), p)+np.dot(np.dot(ay, Cy), p)+np.dot(np.dot(axy, Cxy), p)
                    p = np.array((1., -1., 0.))
                    int_f_1_m_t = np.dot(np.dot(a0, C0), p)+np.dot(np.dot(ax, Cx), p)+np.dot(np.dot(ay, Cy), p)+np.dot(np.dot(axy, Cxy), p)
                
                    J = np.array(np.vstack((np.hstack((0., 1.)), np.hstack((-1., 0.)))))
                    A = int_f_1_m_t
                    B = np.array(np.hstack((int_dfdx_1_m_t_2, int_dfdy_1_m_t_2)))
                    
                    HessianI_i_i[i,:,:] =HessianI_i_i[i,:,:]-J*A+np.dot(np.dot(J, DeltaP).reshape((2,1)), B.reshape((1,2)))
                    A = int_f_t
                    B = np.array(np.hstack((int_dfdx_t2, int_dfdy_t2)))
                    HessianI_i_i[ip,:,:] = HessianI_i_i[ip,:,:]+J* A+np.dot(np.dot(J, DeltaP).reshape((2,1)), B.reshape((1,2)))
                    B = np.array(np.hstack((int_dfdx_t_1_m_t, int_dfdy_t_1_m_t)))
                    HessianI_ip_i[ip,:,:] = HessianI_ip_i[ip,:,:] +(J*int_f_1_m_t+np.dot(np.dot(J, DeltaP).reshape((2,1)), B.reshape((1,2)))).conj().T
                    #Hessian2B[2*ip:2*(ip+1),2*i:2*(i+1)] = Hessian2B[2*ip:2*(ip+1),2*i:2*(i+1)] +(J*int_f_1_m_t+np.dot(np.dot(J, DeltaP).reshape((2,1)), B.reshape((1,2)))).conj().T
                    

    if nargout >= 3 and order==1:
        #i=0,1,0,1,2,3,2,3,4,5,4,5
        #j=0,0,1,1,2,2,3,3,4,4,5,5
        i=np.tile(np.arange(2*N).reshape(N,1,2),(1,2,1))
        j=np.tile(np.arange(2*N).reshape(N,2,1),(1,1,2))            
        Hessian1=sp.coo.coo_matrix((HessianI_i_i.flatten(),(i.flatten(),j.flatten())))
        i=np.mod(np.tile(np.arange(2*N).reshape(N,1,2),(1,2,1))-2,2*N)  
        j=np.tile(np.arange(2*N).reshape(N,2,1),(1,1,2))              
        Hessian2=sp.coo.coo_matrix((HessianI_ip_i.flatten(),(i.flatten(),j.flatten())))
        Hessian=Hessian1+Hessian2+Hessian2.T
        #plt.imshow( Hessian.todense(),interpolation='nearest')
    
    if nargout==1:
        return I
    elif nargout==2:
        return I,dI_dP
    elif nargout==3:
        return I,dI_dP,Hessian
    elif nargout==4:
        M =np.flipud(np.cumsum(np.flipud(S),0))
        return I,dI_dP,Hessian,M
#@autojit    
def _prepare_integral(\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    f,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    L, 
    #cython#np.ndarray[np.float64_t, ndim=1]\
    t):

    #cython#cdef unsigned int K,k
    K = L.shape[1]-1
    
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    f00 = np.empty((K))
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    f11 = np.empty((K))
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    f10 = np.empty((K))
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    f01 = np.empty((K))
    #cython#cdef np.ndarray[np.float64_t, ndim=2]\
    c = 0.5*( L[:,0:K]+L[:,1:K+1.])
    #cython#cdef np.ndarray[np.uint32_t, ndim=2]\
    fc = np.floor(c).astype(np.uint32)
    for k in range(K):
        f00[k] = f[fc[0,k]-1,fc[1,k]-1]
        f11[k] = f[fc[0,k],fc[1,k]]
        f10[k] = f[fc[0,k],fc[1,k]-1]
        f01[k] = f[fc[0,k]-1,fc[1,k]]
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    xc = c[0,:]-fc[0,:]
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    yc = c[1,:]-fc[1,:]
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    tc = 0.5*( t[1:K+1]+t[0:K])
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    Dt = t[1:K+1]-t[0:K]
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    Dy = L[1,1:K+1]-L[1,0:K]
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    Dx = L[0,1:K+1]-L[0,0:K]
    return f00, f01, f10, f11, xc, yc, tc, Dx, Dy, Dt
#@autojit
def _integrePolynomOnLineFragments3(\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    xc,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    yc,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    tc,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    Dx,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    Dy,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    Dt,\
    #cython#Py_ssize_t\
    degree):

    #cython#cdef unsigned int K,j,k
    K = tc.size
    
    #cython#cdef np.ndarray[np.float64_t, ndim=2]\
    c0 = np.empty((K, degree+1),dtype=np.float64) 
    #cython#cdef np.ndarray[np.float64_t, ndim=2]\
    cx = np.empty((K, degree+1),dtype=np.float64) 
    #cython#cdef np.ndarray[np.float64_t, ndim=2]\
    cy = np.empty((K, degree+1),dtype=np.float64) 
    #cython#cdef np.ndarray[np.float64_t, ndim=2]\
    cxy = np.empty((K,degree+1),dtype=np.float64) 
    
    

    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    a = np.empty((3+degree),dtype=np.float64)  
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    Ax = np.empty((2),dtype=np.float64)  
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    Ay = np.empty((2),dtype=np.float64)  
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    Axy = np.empty((3),dtype=np.float64)  
    
    for k in xrange(K):
        # % a(j)=\int_{t_{k}}^{t_{k+1}} t^j dt  
        for j in xrange(0, 3+degree):            
            a[j] = ((tc[k]+Dt[k]/2.)**(j+1)-(tc[k]-Dt[k]/2.)**(j+1))/( j+1) 
            
        Ax[1] = Dx[k]/ Dt[k] # this ratio is constant for a segment , could speedup things!
        Ax[0] = xc[k]-(tc[k]*Dx[k]/ Dt[k])# this seems to be constant for a segment , could speedup things!
        
        Ay[0] = yc[k]-(tc[k]/ Dt[k])* Dy[k]# this seems to be constant for a segment , could speedup things!
        Ay[1] = Dy[k]/ Dt[k]     # this seems to be constant for a segment , could speedup things!   
        #% x(t)=Ax*[1,t]';
        #% y(t)=Ay*[1,t]';       
        Axy[0] = Ax[0]* Ay[0]# this seems to be constant for a segment , could speedup things!
        Axy[1] = Ax[0]* Ay[1]+Ax[1]* Ay[0]# this seems to be constant for a segment , could speedup things!
        Axy[2] = Ax[1]* Ay[1]# this seems to be constant for a segment , could speedup things!
        #%conv(Ax,Ay);
        #% x(t)*y(t)=Axy*[1,t,t^2]';
        #% c_0(k,j)=\int_{tc{k}-Dt{k}/2}^{tc{k}+Dt{k}/2}  t^j dt
        #% c_x(k,j)=\int_{tc{k}-Dt{k}/2}^{tc{k}+Dt{k}/2} x(t)*t^j dt
        #% c_y(k,j)=\int_{tc{k}-Dt{k}/2}^{tc{k}+Dt{k}/2}  y(t)*t^j dt
        #% c_{xy}(k,j)=\int_{tc{k}-Dt{k}/2}^{tc{k}+Dt{k}/2}  x(t)*y(t)*t^j dt
        for j in xrange(degree+1):
            c0[k,j] = a[j]
            #cx[k,j] = np.dot(Ax, a[j:j+2])
            cx[k,j] = Ax[0]*a[j]+Ax[1]*a[j+1]# much faster in cython... 
            #cy[k,j] = np.dot(Ay, a[j:j+2])
            cy[k,j] = Ay[0]*a[j]+Ay[1]*a[j+1]# fmuch faster in cython...
            #cxy[k,j] = np.dot(Axy, a[j:j+3])
            cxy[k,j] = Axy[0]*a[j]+Axy[1]*a[j+1]+Axy[2]*a[j+2]# much faster in cython... 
            
    return c0,cx,cy,cxy
#@autojit
def _integreLeftLineFragements(\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    L,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    F,\
    #cython#int\
    order,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    S):
 

    # integre f in the trapezoid at the left of a subpixel
    # line fragment
    # example :
    # F=rand(10,10);
    # L=[5.1,5.7;6.04,6.37]
    # integreLeftLineFragements(L,[],F,'bilinear')
    
        
    
    #cython#cdef unsigned int k
    #cython#cdef double c,fc,e,h,I,yc,xc,yce,xce,w,c00,c01,c10,c11,f00,f01,f10,f11
    #cython#cdef int xfc,yfc
    I = 0.
    
 
    #cython#cdef int Nv,Nh    
    
    if order==0:
        for k in range(L.shape[1]-1): 
            
            xc=0.5*( L[0,k]+L[0,k+1])
            yc = 0.5*( L[1,k]+L[1,k+1]) 
            xfc=int(floor(xc))
            yfc=int(floor(yc))  
            e = xc-xfc
            
            h = L[1,k+1]-L[1,k]
            if h==0:
                continue
            
            if e>0:
                I += (h*e)*(F[xfc,yfc])     
                
                if S.size>0:
                    
                    S[xfc,yfc] = S[xfc,yfc]+h* e
                
                   
            if xfc > 0.:
                I += h*( 1.-e)*(F[xfc-1,yfc])
                if S.size>0 :
                    
                    S[(xfc)-1,yfc] = S[(xfc)-1,yfc]+h*( 1.-e)
                
                
            
            
            
    elif order==1:
        for k in range(L.shape[1]-1):           
            xc=0.5*( L[0,k]+L[0,k+1])
            yc = 0.5*( L[1,k]+L[1,k+1])   
            
            xfc=int(floor(xc))
            yfc=int(floor(yc))            
            yce = yc-yfc
            xce = xc-xfc  
            
            h = L[1,k+1]-L[1,k]
            w = L[0,k+1]-L[0,k]
            if h==0:
                continue
            
            #I += h* (yce* (F[xfc-1,yfc]-0.5*( F[0,yfc])-0.5*( f01))+(1.-yce)* (F[xfc-1,yfc-1]-0.5*( F[0,yfc-1])-0.5*( F[xfc-1,yfc-1]-F[xfc-2,yfc-1])))
            f01 = F[xfc-1,yfc]-F[xfc-2,yfc]  
            f00 = F[xfc-1,yfc-1]-F[xfc-2,yfc-1]
            I += h* (yce* (F[xfc-1,yfc]-0.5*( f01))+(1.-yce)* (F[xfc-1,yfc-1]-0.5*( f00)))
            
           
    
            
            c11 = (1./24.)* h*(2.*h*xce*w+yce*w**2.+12.*yce*xce**2.)
            c10 = -c11+(1./24.)*h*(w**2.+12.*xce**2.)
            c01 = -c11+h*(xce*yce+(1./12.)*w*h)
            c00 = -c10+h*(xce*(1.-yce)-(1./12.)*w*h)
            
            if w!=0:
                f11 = F[xfc,yfc]-F[xfc-1,yfc]
                f10 = F[xfc,yfc-1]-F[xfc-1,yfc-1]          

        
                I += f00* c00+f10* c10+f01* c01+f11* c11
            if S.size>0:
                if h!=0:
                    S[xfc-2,yfc-1] = S[xfc-2,yfc-1] -c00+0.5*h*(1-yce)
                    S[xfc-2,yfc  ] = S[xfc-2,yfc  ] -c01+0.5*h*yce
                    S[xfc-1,yfc-1] = S[xfc-1,yfc-1] +c00-c10+h*(1-yce)-0.5*h*(1-yce)
                    S[xfc-1,yfc  ] = S[xfc-1,yfc  ] -c11+c01+h*yce-0.5*h*yce
                if w!=0:
                    S[xfc  ,yfc-1] = S[xfc  ,yfc-1] +c10
                    S[xfc  ,yfc  ] = S[xfc  ,yfc  ] +c11
                
                
                #% can remove the two line below because it cancel out when the
                #% polygon is closed
                #%S(1,fc(2)+1)=S(1,fc(2)+1)               -0.5*yce*h;
                #%S(1,fc(2))  =S(1,fc(2))                 -0.5*h*(1-yce);
        
            
            
        
    else:
        error('unkown interpolation method')
        
    

   
    return I

#@autojit    
def _computeGradientBilinear(\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    P,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    f00,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    f01,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    f10,\
    #cython#np.ndarray[np.float64_t, ndim=1]\
    f11,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    C0,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    Cx,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    Cy,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    Cxy,\
    #cython#int\
    i,\
    #cython#int\
    ip,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    dI_dP):
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    DeltaP = P[ip]-P[i]
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    ay = f01-f00
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    axy = f11-f10+f00-f01
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    ax = f10-f00
    #cython#cdef np.ndarray[np.float64_t, ndim=1]\
    a0 = f00                    
    
    tmp=np.dot(a0, C0)+np.dot(ax, Cx)+np.dot(ay, Cy)+np.dot(axy, Cxy)
    #p = np.array(np.vstack((0., 1.)))
    # int_f_t = np.dot(tmp, p) 
    #cython#cdef double int_f_t,int_f_1_m_t
    int_f_t=tmp[1]    
    #p = np.array(np.vstack((1., -1.)))
    #int_f_1_m_t =np.dot(tmp, p)
    int_f_1_m_t=tmp[0]-tmp[1]
    #dI_dP[ip] = dI_dP[ip]+np.dot(np.array(np.vstack((np.hstack((0., 1.)), np.hstack((-1., 0.))))), DeltaP)*int_f_t
    #dI_dP[i] = dI_dP[i]+np.dot(np.array(np.vstack((np.hstack((0., 1.)), np.hstack((-1., 0.))))), DeltaP)* int_f_1_m_t
    dI_dP[i,0] = dI_dP[i,0] + DeltaP[1]* int_f_1_m_t    
    dI_dP[i,1] = dI_dP[i,1] - DeltaP[0]* int_f_1_m_t      
    dI_dP[ip,0] = dI_dP[ip,0] + DeltaP[1]*int_f_t
    dI_dP[ip,1] = dI_dP[ip,1] - DeltaP[0]*int_f_t    
#@autojit    
def _computeGradientConst(\
     #cython#np.ndarray[np.float64_t, ndim=2]\
     F,\
     #cython#np.ndarray[np.float64_t, ndim=2]\
     P,\
     #cython#np.ndarray[np.float64_t, ndim=2]\
     L,\
     #cython#np.ndarray[np.float64_t, ndim=1]\
     t,\
     #cython#int\
     i,\
     #cython#int\
     ip,\
     #cython#np.ndarray[np.float64_t, ndim=2]\
     dI_dP):   
   
   
  
    
    #cython#cdef double xc,yc
    #cython#cdef unsigned int xfc,yfc
    #cython#cdef unsigned int k    
    #cython#cdef double  DeltaPx, DeltaPy
    #cython#cdef double DeltaT    
    #cython#cdef double u,v,tmp,tc    
    
    DeltaPx = P[ip,0]-P[i,0]                
    DeltaPy = P[ip,1]-P[i,1]
    
  

    u=0
    v=0
    for k in range(L.shape[1]-1):
        xc=0.5*( L[0,k]+L[0,k+1])
        yc = 0.5*( L[1,k]+L[1,k+1]) 
        xfc=int(floor(xc))
        yfc=int(floor(yc)) 
        DeltaT=t[k+1]-t[k]
        tc=0.5*(t[k+1]+t[k]) 
        tmp = DeltaT*(F[xfc,yfc]-F[xfc-1,yfc])  
        u+=tmp
        v+=tc*tmp      
   
    dI_dP[i,0] += (u-v)*DeltaPy
    dI_dP[i,1] -= (u-v)* DeltaPx  
    dI_dP[ip,0] += v*DeltaPy
    dI_dP[ip,1] -= v* DeltaPx

def _test():
    import doctest
    doctest.testmod()

def _test2(display=False):   
    from xorshift import xorshift
    rd=xorshift()      
   
    Im = ImageWithIntegral(rd.random(50, 50))
    N = 5
    P = rd.random(2, N).T*49.+2.# TO DO : if we get closer tho the borders of the image it seems like the gradient becomes false
    
    [Ic, dIc_dP, Hessianc, Mc] = integreWithinPolygon(Im,P, 'const',nargout=4)
    [Ib, dIb_dP, Hessianb, Mb] = integreWithinPolygon(Im,P, 'bilinear',nargout=4)
    
    if display:
        plt.ion()
        plt.figure
        plt.imshow(Mc.conj().T,cmap = plt.get_cmap('gray'))
        plt.show()
    
        plt.figure
        plt.imshow(Mb.conj().T,cmap = plt.get_cmap('gray'))
        plt.show()    
        
    assert abs( Mc.flatten().dot(Im.values.flatten())- Ic)<1e-10
    assert abs( Mb.flatten().dot(Im.values.flatten())-Ib)<1e-10
    
    assert(Ic==47.561991223913537)
    assert(Ib==45.969218974912508)
    
    
    
    func = lambda x: integreWithinPolygon(Im,x,  'const',nargout=2)
    assert(verif_gradient(func, P,verbose=False))
    func = lambda x: integreWithinPolygon(Im,x,  'bilinear',nargout=2)
    assert(verif_gradient(func, P,verbose=False))
    I, dI_dP, Hessian = integreWithinPolygon(Im,P, 'bilinear',nargout=3)
    Hnum = numHessian(func, P, 1e-7)
    
    abs_angle = np.reshape(np.multiply(((Hessian.todense()).flatten(1) != 0.),((np.abs((np.mod(np.arctan2(Hnum.flatten(1), (Hessian.todense()).flatten(1)), np.pi)-np.pi/4.))))  ),Hessian.shape)
    assert (np.max(abs_angle.flatten())<5e-5)
    

  

if __name__ == '__main__':
    
    _test2(display=False)
    _test()
   
   
    

