
import numpy as np
import scipy
import scipy.sparse as sp


from interp_bilinear import *
from interp_bicubic import *
from clipSegmentOnGrid import *
from verif_gradient import *
from numHessian import *
from xorshift import *

def matdiv(a,b):
    return a/b

def sparse(m,n):
    return  sp.lil_matrix((m,n),dtype=np.float)

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def integreOnSegments(P, E, f, interp='bicubic',nargout=1):

    # [I,dI_dP,M]=integreOnSegments(P,E,f,interp)
    # Integrates an interpolation of f alongs mutliple segments
    #
    # INPUT :
    # P : vertices, matrix of size 2 by Np
    # E : edges as pairs of vertices index, matrix of size 2 by Ne
    # f : the discretized 2D function beeing integrated
    # interp: interpolation sheme ('const','bilinear','bicubic')	
    # OUTPUT
    # I     : the integral
    # dI_dP : the derivative dI/dP (i.e forces on vertices)
    # M     : the image of pixels weights such that I==sum(M(:).*f(:))
    #
    # coded by Martin de la Gorce

    
    if P.shape[1] != 2:
        raise 'P should by N by 2'   
    
    if E.shape[1]!= 2:
        raise 'E should by N by 2'
    

    Np = P.shape[0]
    Ne = E.shape[0]
    I = 0
 
    if nargout == 4.:
        M = np.zeros((f.shape[0], f.shape[1]) )   
    else: 
        M = np.zeros((0,0) )     
   
    for k in np.arange(1, Ne+1):
        i = E[k-1,0]+1
        ip = E[k-1,1]+1
        L = clipSegmentOnGrid(P[i-1,0], P[i-1,1], P[ip-1,0], P[ip-1,1],nargout=1)
        L=L.T
        Is = integreOnLineFragements(L, f, interp,M)    
        I = I+Is
        
   
    
    if nargout >= 2:
        if interp.lower()== 'const':
            dI_dP = np.zeros(( Np,2))
            HessianI = np.array([])
            print('not yet coded')
        elif interp.lower()==  'bilinear':
            dI_dP = np.zeros(( Np,2))
            HessianI = np.array([])
            for k in np.arange(1., Ne+1):

    
                i = E[k-1,0]+1
                ip = E[k-1,1]+1
                L = clipSegmentOnGrid(P[i-1,0], P[i-1,1], P[ip-1,0], P[ip-1,1],nargout=1)
                L=L.T
                Len = np.sqrt(np.sum(((L[:,0:-1]-L[:,1:])**2.), 0))
                DeltaP = P[ip-1]-P[i-1]
                #% Ltot=norm(DeltaP);
                if np.abs(DeltaP[0]) > np.abs(DeltaP[1]):
                    u = (L[0,:]-P[ip-1,0])/ (P[i-1,0]-P[ip-1,0])
                else:
                    u = (L[1,:]-P[ip-1,1])/( P[i-1,1]-P[ip-1,1])
    
                for k2 in np.arange(1, L.shape[1]):
   
                    
                    c = 0.5*( L[:,k2-1]+L[:,k2])
                    fc = np.floor(c)
                    f00 = f[fc[0]-1,fc[1]-1]
                    f11 = f[fc[0],fc[1]]
                    f10 = f[fc[0],fc[1]-1]
                    f01 = f[fc[0]-1,fc[1]]
                    xa = L[0,k2-1]-fc[0]
                    ya = L[1,k2-1]-fc[1]
                    xb = L[0,k2]-fc[0]
                    yb = L[1,k2]-fc[1]
                    ua = u[k2-1]
                    ub = u[k2]                    
       
                    
                    sx=(1./3)*((ya-yb)*f11+(-ya+yb)*f10-(ya-yb)*f01-(-ya+yb)*f00)*(ua-ub)+0.5*(yb*f11+(1-yb)*f10-yb*f01-(1-yb)*f00)*(ua-ub)+0.5*((ya-yb)*f11+(-ya+yb)*f10-(ya-yb)*f01-(-ya+yb)*f00)*ub+(yb*f11+(1-yb)*f10-yb*f01-(1-yb)*f00)*ub
                    sy=(1./3)*((xa-xb)*f11-(xa-xb)*f10+(-xa+xb)*f01-(-xa+xb)*f00)*(ua-ub)+0.5*(xb*f11-xb*f10+(1-xb)*f01-(1-xb)*f00)*(ua-ub)+0.5*((xa-xb)*f11-(xa-xb)*f10+(-xa+xb)*f01-(-xa+xb)*f00)*ub+(xb*f11-xb*f10+(1-xb)*f01-(1-xb)*f00)*ub
                    dI_dP[i-1] = dI_dP[i-1]+np.array((sx,sy))* Len[k2-1]
                    ua = 1.-ua
                    ub = 1.-ub
                    sx=(1./3)*((ya-yb)*f11+(-ya+yb)*f10-(ya-yb)*f01-(-ya+yb)*f00)*(ua-ub)+0.5*(yb*f11+(1-yb)*f10-yb*f01-(1-yb)*f00)*(ua-ub)+0.5*((ya-yb)*f11+(-ya+yb)*f10-(ya-yb)*f01-(-ya+yb)*f00)*ub+(yb*f11+(1-yb)*f10-yb*f01-(1-yb)*f00)*ub
                    sy=(1./3)*((xa-xb)*f11-(xa-xb)*f10+(-xa+xb)*f01-(-xa+xb)*f00)*(ua-ub)+0.5*(xb*f11-xb*f10+(1-xb)*f01-(1-xb)*f00)*(ua-ub)+0.5*((xa-xb)*f11-(xa-xb)*f10+(-xa+xb)*f01-(-xa+xb)*f00)*ub+(xb*f11-xb*f10+(1-xb)*f01-(1-xb)*f00)*ub
                    dI_dP[ip-1] = dI_dP[ip-1]+np.array((sx,sy))* Len[k2-1]

                  
                Is = integreOnLineFragements(L, f, 'bilinear', np.zeros((0,0)))
                tmp = np.dot(DeltaP, Is)/np.sum((DeltaP**2))
                dI_dP[i-1] = dI_dP[i-1]-tmp
                dI_dP[ip-1] = dI_dP[ip-1]+tmp

         
            
        else:
            raise ('unkown interpolation method')
            
            
        
        
        
    if nargout==1:
        return I
    if nargout==2:
        return I,dI_dP
    if nargout==3:
        return I, dI_dP, HessianI
    if nargout==4:    
        return I, dI_dP, HessianI, M
    
    
def integreOnLineFragements(L, f, interp,M):

    # integre f along a subpixel  line fragment
    
 
    
    I = 0
    Len = np.sqrt(np.sum(((L[:,0:-1]-L[:,1:])**2), 0))
    if interp.lower()== 'const':
        for k in np.arange(1,L.shape[1]):
            c = 0.5*( L[:,k-1]+L[:,k])
            fc = np.floor(c)
            I = I+Len[k-1]*f[fc[0]-1,fc[1]-1]
            if M.size>0:
                M[fc[0]-1,fc[1]-1] = M[fc[0]-1,fc[1]-1]+Len[k-1]
            
    
    elif interp.lower()==  'bilinear':     
      
        for k in np.arange(1, L.shape[1]):
            c   = 0.5*( L[:,k-1]+L[:,k])
            fc  = np.floor(c)
            f00 = f[fc[0]-1,fc[1]-1]
            f11 = f[fc[0]   ,fc[1]]
            f10 = f[fc[0]  ,fc[1]-1]
            f01 = f[fc[0]-1,fc[1]]
            xa  = L[0,k-1]-fc[0]
            ya  = L[1,k-1]-fc[1]
            xb  = L[0,k]-fc[0]
            yb  = L[1,k]-fc[1] 
            xc = c[0]-fc[0]
            yc = c[1]-fc[1]
            r  = (1./12.)*( xa-xb)*( ya-yb)
            c00 = (1-xc)*( 1-yc)+r
            c11 = (xc* yc)+r
            c01 = ((1-xc)* yc)-r
            c10 = (xc*( 1-yc))-r
            s = f00* c00+f10* c10+f01* c01+f11* c11
            I = I+Len[k-1]* s
            if M.size>0:
                M[fc[0]-1,fc[1]-1] = M[fc[0]-1,fc[1]-1]+ Len[k-1]* c00
                M[fc[0],  fc[1]-1] = M[fc[0],  fc[1]-1]+ Len[k-1]* c10
                M[fc[0]-1,  fc[1]] = M[fc[0]-1,  fc[1]]+ Len[k-1]* c01
                M[fc[0],    fc[1]] = M[fc[0],    fc[1]]+ Len[k-1]* c11
            
            
            
        
    elif interp.lower()==  'bicubic':
        raise 'not yet coded'
        
    else:
        np.disp(interp)
        raise 'unkown interpolation method'
        
 
    
    return I


if __name__ == '__main__':

    # Local Variables: nH, x, dI2_dP, E, func, f, I1, d2I4, I3, I2, I4, d2I3, N, M4, P, T, M3, dI1_dP, dI3_dP, d2I2, dI4_dP
    # Function calls: rand, verif_gradient, figure, sum, imshow, integreOnSegments, numHessian, full, test, delaunay
    rd=xorshift()    
    f = rd.random(150, 150)
    N = 10
    #%E=[1,1,1;2,3,4]
    P = rd.random(2, N).T*147.+2.
    
    #from scipy.spatial import Delaunay    
    #T = Delaunay(P.T).vertices 
    #E = np.vstack((T[:,0:2], T[:,1:3], T[:,::-2])).T
    
    E= numpy.array([[i,i+1] for i in range(N-1)])
    I1, dI1_dP = integreOnSegments(P, E, f, 'bilinear', nargout=2)
    I4, dI4_dP, d2I4, M4 = integreOnSegments(P, E, f, 'bilinear',nargout=4)
    I3, dI3_dP, d2I3, M3 = integreOnSegments(P, E, f, 'const',nargout=4)
    plt.ion()
    plt.figure()
    plt.imshow(M3.conj().T,cmap = plt.get_cmap('gray'))
    plt.show()
    plt.figure
    plt.imshow(M4.conj().T,cmap = plt.get_cmap('gray'))
    plt.show()
    print(I3)
    print(float(M3.flatten().dot(f.flatten())))
    print(I4)
    print(float(M4.flatten().dot(f.flatten())))
    func = lambda x: integreOnSegments(x, E, f, 'bilinear', nargout=2)
    verif_gradient(func, P)
    func = lambda x: integreOnSegments(x, E, f, 'bilinear', nargout=2)
    verif_gradient(func, P)   
     
   
    
