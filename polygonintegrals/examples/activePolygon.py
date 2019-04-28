from polygonintegrals.xorshift import *
from scipy.interpolate import interp1d 
from polygonintegrals.integreWithinPolygon import integreWithinPolygon,ImageWithIntegral
from polygonintegrals.integreOnPolygonBoundary import integreOnPolygonBoundary
import matplotlib.pylab as plt
import numpy as np
import scipy
import optimizers
import time
from optimizers import optimTimer

def subsamplePoly(x,y,n):
    """this function eases the process of subsampling a polygon"""
       
    x=np.array(x)
    y=np.array(y)
    xc=np.hstack((x,x[0]))
    yc=np.hstack((y,y[0]))

    t=np.arange(len(x)+1).astype(float)/len(x)

    fx = interp1d(t, xc, kind='linear')
    fy = interp1d(t, yc, kind='linear')

    t2=np.arange(n).astype(float)/n
    nx=fx(t2)
    ny=fy(t2)
    return np.vstack((nx,ny)).T        

def differencesMatrix1D(N):
    """ creates the sparse circulant matrix of size N-1 by N that allows to compute
        the difference between successive elements of a vector through a simple multiplication
        differencesMatrix1D(5).todense() should give        
        matrix([[-1.,  1.,  0.,  0.,  0.],
                [ 0., -1.,  1.,  0.,  0.],
                [ 0.,  0., -1.,  1.,  0.],
                [ 0.,  0.,  0., -1.,  1.],
                [ 1.,  0.,  0.,  0., -1.]
                ])
    """
    #TO DO create the matrix using sparse.diags with the optional shape argument
    diagonals=[1,-np.ones((N)),np.ones((N-1))]
    D=scipy.sparse.diags(diagonals, [-N+1,0,1],shape=(N,N))
    return D  


def plotClosedPolygon(x,ax_image=None):
    P=x.reshape(-1,2)
    ids=np.mod(np.arange(len(P)+1),len(P))
    if not ax_image is None:
        ax_image.plot(P[ids,1],P[ids,0]-1,'o-',linewidth=3,markersize=10)
    else:
        plt.plot(P[ids,1],P[ids,0]-1,'o-',linewidth=3,markersize=10)

class ActivePolygon():
    def __init__(self,ImRegion,ReferenceShape,coefRegu):
        N=len(ReferenceShape)
        diffM=scipy.sparse.kron(differencesMatrix1D(N),scipy.sparse.eye(2))
        self.reguM=(diffM.T*diffM).tocsr()
        self.coefRegu=coefRegu
        self.ImRegion=ImRegion        
        self.x0=ReferenceShape.flatten()
        
    def Iregu(self,x):  
        return 0.5*(x-self.x0).dot(self.reguM*(x-self.x0))
        
    def Gredu(self,x):
        return self.reguM*(x-self.x0)
        
 
        
        
        
    def draw(self,ax_image,x):
        plotClosedPolygon(x, ax_image)
        
    def funcNoJac(self,x):  
        
        IRegion=integreWithinPolygon(self.ImRegion,x.reshape(-1,2),  'bilinear',nargout=1)  
        #IEdges =integreOnPolygonBoundary(x.reshape(-1,2),self.ImEdges, 'bicubic', nargout=1) 
        IEdges=0
        I=IRegion+IEdges+self.coefRegu*self.Iregu(x)
        return I

    #@profile      
    def funcWithJac(self,x):
        IRegion,GRegion =integreWithinPolygon(self.ImRegion,x.reshape(-1,2),  'bilinear',nargout=2) 
        #IEdges, GEdges  =integreOnPolygonBoundary(x.reshape(-1,2),self.ImEdges, 'bicubic',  nargout=2)
        IEdges, GEdges=(0,0)
        I=IRegion+IEdges+self.coefRegu*self.Iregu(x)        
        G=GRegion.flatten()+GEdges+self.coefRegu*self.Gredu(x)
        return I,G

    def funcWithJacAndHess(self,x):
        IRegion,GRegion,HRegion  =integreWithinPolygon(self.ImRegion,x.reshape(-1,2),  'bilinear',nargout=3)
        #IEdges, GEdges, HEdges   =integreOnPolygonBoundary(x.reshape(-1,2),self.ImEdges, 'bicubic',nargout=3)
        IEdges, GEdges, HEdges=(0,0,0)
        Hregu=self.reguM
        H=HRegion+HEdges+self.coefRegu*Hregu
        I=IRegion+IEdges+self.coefRegu*self.Iregu(x)     
        G=GRegion.flatten()+GEdges+self.coefRegu*self.Gredu(x)
        return I,G,H  

    def funcWithJacAndApproxHess(self,x):
        IRegion,GRegion  =integreWithinPolygon(self.ImRegion,x.reshape(-1,2),  'bilinear',nargout=2)
        #IEdges, GEdges   =integreOnPolygonBoundary(x.reshape(-1,2),self.ImEdges, 'bicubic', nargout=2)
        IEdges, GEdges=(0,0)
        Hregu=self.reguM      
        I=IRegion+IEdges+self.coefRegu*self.Iregu(x)
        G=GRegion.flatten()+GEdges+self.coefRegu*self.Gredu(x)
        return I,G,Hregu     

    def hessFunc(self,x):
        Idata,Gdata,Hdata=integreWithinPolygon(self.ImRegion,x.reshape(-1,2),  'bilinear',nargout=3)
        #IEdges, GEdges, HEdges   =integreOnPolygonBoundary(x.reshape(-1,2),self.ImEdges,   'bicubic',nargout=3)
        IEdges, GEdges, HEdges=(0,0,0)
        Hregu=self.reguM
        H=HRegion+HEdges+self.coefRegu*Hregu
        return H.todense()        
        
        

      
    

#@profile  
def test():
    plt.ion()
    rd=xorshift()      
    im=rd.random(200, 200)-0.5
    #use a gausian mixture , but need to be able to weight the samples on the boundary, could 
	# modify def _estimate_gaussian_parameters in https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/mixture/gaussian_mixture.py
	# to get an optional weights vector 
        #or can adapt https://github.com/stober/gmm/blob/master/src/gmm.py
    #it will be slow as we need to update thet
    # mixture using all the pixels in the image
    # there is no obvious way to udpatne the gaussian texture model by considering only the new pixel added or the pixel 
    # removed from the region (maybe using a hierarchical gaussian mixture, or by updating only)a subset of the 
    # variables (responsability weights for the new points and the gaussian parameters ? ) but then harder to perform newton optimization
    # use color histogram instead ? or use a gausian mixture for the entire image and then keep the gaussians fixed
   
    import skimage.filter
    im = skimage.filter.gaussian_filter(im, 3)
    
    import sklearn.mixture.gmm as gmm
    
    
    
    from  scipy.misc import imread
    
   
    

    Rectangle=np.ones((200,200))
    Rectangle[80:180,50:150]=0

    #I,J=np.meshgrid(np.arange(200),np.arange(200))
    #gauss=np.exp(-((I-140.)**2+(J-100.)**2)/(50**2))-0.5
    im=im+0.05*(Rectangle-0.5)
    fig_image=plt.figure(figsize=(8,8)) 
    ax_image=plt.subplot(111)    
    plt.xlim([0,im.shape[1]])
    plt.ylim([im.shape[0],0])
    ax_image.get_xaxis().set_visible(False)
    ax_image.get_yaxis().set_visible(False)
    plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01)
     
    ax_image.imshow(im,cmap=plt.cm.Greys_r,interpolation='nearest') 

   
    ImRegion = ImageWithIntegral(im)
   
    fig_curve=plt.figure()
    ax_curve=plt.subplot(121)    
    ax_dist=plt.subplot(122) 
    ax_dist.set_yscale('log')
    #plt.imshow(im)
    
    N = 20
    x=[50,im.shape[1]-50,im.shape[1]-50,50]
    y=[50,50,im.shape[0]-50,im.shape[0]-50]
    ReferenceShape = subsamplePoly(x, y, N)+np.random.rand(N,2)*0.01  # need to add some noisy because my function to clip segments on pixels nis not very robust to degenerated cases
    x0=ReferenceShape.flatten()
    activePolygon=ActivePolygon(ImRegion,ReferenceShape,coefRegu=0.5)      
    activePolygon.draw(ax_image,x0)
    plt.draw()    
    plt.show()
    
    
    bounds=np.ones((2*N,2))
    bounds[::2,1]=im.shape[0]-1
    bounds[1::2,1]=im.shape[1]-1
    
    
  
    
    ot=optimTimer(None,activePolygon.funcNoJac,ax_curve,ax_dist)
    res=scipy.optimize.minimize(activePolygon.funcWithJac, x0.copy(), method='CG',jac=True,callback=ot.callback,options={'maxiter':100})    
    activePolygon.draw(ax_image,res['x'])
    ot.setOptim(res['x'])
    ot.stats('CG')
    plt.draw()
    plt.show()     
    
    
    ot.clear()
    res=optimizers.simpleTrustRegion(x0.copy(),activePolygon.funcNoJac,activePolygon.funcWithJacAndHess,radius=20,maxradius=20,tikonow=0.01,max_duration=10,max_iter=100,verbose=False,callback=ot.callback)
    activePolygon.draw(ax_image,res['x'])
    ot.stats('simpleTrustRegion')
    plt.draw()
    plt.show()
   
    
    
    ot.clear()
    res=optimizers.simpleTrustRegion(x0.copy(),activePolygon.funcNoJac,activePolygon.funcWithJacAndApproxHess,radius=20,maxradius=20,tikonow=0.01,max_duration=10,max_iter=100,verbose=False,callback=ot.callback)
    activePolygon.draw(ax_image,res['x'])
    ot.stats('simpleTrustRegion Approx H')
    plt.draw()
    plt.show()
   
    ot.clear()
    res=optimizers.simpleGradientDescent(x0.copy(),activePolygon.funcNoJac,activePolygon.funcWithJac,max_duration=10,max_iter=100,verbose=False,callback=ot.callback)
    activePolygon.draw(ax_image,res['x'])
    ot.stats('gradient descent')
    plt.draw()
    plt.show()
  
    
    

    
    ot=optimTimer(None,activePolygon.funcNoJac,ax_curve,ax_dist)
    res=scipy.optimize.minimize(activePolygon.funcWithJac, x0.copy(), method='TNC',bounds=bounds,jac=True,callback=ot.callback,options={'maxiter':100})    
    activePolygon.draw(ax_image,res['x'])
    ot.setOptim(res['x'])
    ot.stats('TNC')
    plt.draw()
    plt.show()  







if __name__ == '__main__':

    test()
