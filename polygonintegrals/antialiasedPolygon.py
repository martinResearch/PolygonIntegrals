from __future__ import absolute_import
import numpy as np
from .integreWithinPolygon import _integreLeftLineFragements
from .clipSegmentOnGrid import clipSegmentOnGrid
from .polygonClipping import clip

def drawAntialiasedPolygon(Image,P,color,interp=0,use_xy=True):

    if use_xy:
        P=P[:,[1,0]]
    P=P.astype(np.float) 
        
    if interp.lower()== 'const':
        order=0
    elif interp.lower()== 'bilinear':
        order=1
    else:
        raise Exception('unkown interpolation method')            
        
    if  order==0:
        if np.any(P[:,0]<0)  or np.any(P[:,0]>Image.shape[0]) or np.any(P[:,1]<0) or np.any(P[:,1]>Image.shape[1]):
            clipPolygon=np.array([[0,0],[Image.shape[0],0],[Image.shape[0],Image.shape[1]],[0,Image.shape[1]]],dtype=np.float)
            P=clip(P, clipPolygon)
            P=np.maximum(P,0)#deal with roundoff errors
            P[:,0]=np.minimum(P[:,0],Image.shape[0])
            P[:,1]=np.minimum(P[:,1],Image.shape[1])
            if len(P)==0:
                return
    else:
        if np.any(P[:,0]<2)  or np.any(P[:,0]>Image.shape[0]-1) or np.any(P[:,1]<2) or np.any(P[:,1]>Image.shape[1]-1):
            clipPolygon=np.array([[3,3],[Image.shape[0]-1,2],[Image.shape[0]-1,Image.shape[1]-1],[2,Image.shape[1]-1]],dtype=np.float)
            P=clip(P, clipPolygon)
            P=np.maximum(P,2)#deal with roundoff errors
            P[:,0]=np.minimum(P[:,0],Image.shape[0]-1)
            P[:,1]=np.minimum(P[:,1],Image.shape[1]-1)
            if len(P)==0:
                return    
                
    
            
        
       


    S = np.zeros((Image.shape[0], Image.shape[1]) )
    F = np.zeros((Image.shape[0], Image.shape[1]) )
    N = P.shape[0]
    for i in range(N):
        ip = (i+1)% N
       
        L = clipSegmentOnGrid(P[i,0], P[i,1], P[ip,0], P[ip,1],1,False)
       
        _integreLeftLineFragements(L.T, F, order,S)
    M =np.flipud(np.cumsum(np.flipud(S),0))
    if np.any(M<-1e-6): 
        M=-M
    if np.any(M<-1e-6): 
        BaseException('you need to provide a non self-intersecting polygon')
    if np.ndim(Image)==3:
        Image[:,:,:]=Image*(1-M)[:,:,None]+M[:,:,None]*np.array(color)[None,None,:]
    elif np.ndim(Image)==2:
        Image[:,:]=Image*(1-M)+M*color
    else:
        BaseException('you image should be of dimension 2 or 3 (MxM or MxNx3)')
        
    

if __name__=='__main__':
    import matplotlib.pylab as plt
    im=np.zeros((60,60,3))    
    poly=np.array([[10,10],[14,50],[50,31.5]])
    drawAntialiasedPolygon(im,poly, [1,0,0], 'bilinear')
    plt.subplot(1,2,1)
    plt.imshow(im,interpolation='none')
    
   
    im=np.zeros((60,60))    
    poly=np.array([[-5,-10],[40,10],[100,100],[5,40]])
    
    drawAntialiasedPolygon(im,poly, 0.7, 'bilinear')
    plt.subplot(1,2,2)
    plt.imshow(im,cmap='Greys_r',interpolation='none')   
    plt.show()

    P=np.load('P.npy')
    im=np.zeros((29,52))
    drawAntialiasedPolygon(im,P, 0.7, 'bilinear')
