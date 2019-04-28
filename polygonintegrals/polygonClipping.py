#cython: profile=True
#cython: boundscheck=False    
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython#from libc.math cimport floor,ceil,abs
#cython#cimport numpy as np
#cython#cimport cython
# code from https://rosettacode.org/wiki/Sutherland-Hodgman_polygon_clipping
# The   Sutherland-Hodgman clipping algorithm   finds the polygon that is the intersection between an arbitrary polygon (the “subject polygon”) and a convex polygon (the “clip polygon”).
import numpy as np

def clipOnside(#cython#np.ndarray[np.float64_t, ndim=2]\
    subjectList,
    #cython#float\
    cp10,
    #cython#float\
    cp11,
    #cython#float\
    cp20,
    #cython#float\
    cp21):
    #cython#cdef  int einside,sinside,idclip,ns,nc,nbin,nbout
    #cython#cdef  float dc0,dc1,dp0,dp1,n1,n2,n3,e0,e1,s0,s1,interx,intery
    #cython#cdef np.ndarray[np.float64_t, ndim=2]\
    outList = np.empty((2*subjectList.shape[0],2),dtype=np.float)
    
    dc0 = cp10-cp20
    dc1 = cp11-cp21 
    n1 = cp10 * cp21 - cp11 * cp20   

    nbout=0
    nbin=subjectList.shape[0]
    s0=subjectList[nbin-1,0]
    s1=subjectList[nbin-1,1]
    
    
    for idin in range(nbin):
        e0 = subjectList[idin,0]  
        e1 = subjectList[idin,1] 
        einside=(cp20-cp10)*(e1-cp11) > (cp21-cp11)*(e0-cp10)
        sinside=(cp20-cp10)*(s1-cp11) > (cp21-cp11)*(s0-cp10)
        if einside!=sinside:
            dp0 = s0-e0
            dp1= s1-e1                 
            n2 = s0 * e1 - s1 * e0  
            n3 = 1.0 / (dc0 * dp1 - dc1 * dp0)
            interx=(n1*dp0 - n2*dc0) * n3
            intery=(n1*dp1 - n2*dc1) * n3                              
            outList[nbout,0]=interx
            outList[nbout,1]=intery
            nbout+=1 
        if einside:
            outList[nbout,0]=e0
            outList[nbout,1]=e1
            nbout+=1 
        s0=e0
        s1=e1
    return outList[:nbout,:]
   

def clip( \
    #cython#np.ndarray[np.float64_t, ndim=2]\
    subjectPolygon,\
    #cython#np.ndarray[np.float64_t, ndim=2]\
    clipPolygon):
        
    #cython#cdef  float cp10,cp11,cp20,cp21
    #cython#cdef  int idclip,nc

    
    
    nc=len(clipPolygon)
    cp10 = clipPolygon[nc-1,0]
    cp11 = clipPolygon[nc-1,1]
  
    workingList=subjectPolygon

    for idclip in range(nc):
        cp20 = clipPolygon[idclip,0]
        cp21 = clipPolygon[idclip,1]  
        
        workingList=clipOnside( workingList,cp10,cp11,cp20,cp21)
        if len(workingList)==0:
            return workingList
        cp10 = cp20
        cp11 = cp21
        
    return(workingList)
