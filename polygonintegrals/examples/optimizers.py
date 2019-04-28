import time
import scipy
import numpy as np

def simpleTrustRegion(x0,funcNoJac,funcWithJacAndHess,radius,maxradius,tikonow,max_duration,max_iter,tol=1e-6,verbose=True,maxiterCG=20,callback=None,bounds=None):
    """Simple and very naive implementation of a trust region method
    the functions in scipy.optimize do not seem to exploit the full information in the hessian at each iteration
    but require only to be able to compute one hessian-vector products at each iteration.
    Furthermore getting the hessian requires an addition call, which may created wasted computation"""
    curve=[]
    times=[]  

    radius=20
    maxradius=20
    tikonow=0.01    
    x=x0.copy()
    start=time.clock()

    for iter in range(max_iter):
        I,G,H = funcWithJacAndHess(x)
        if not callback is None:
            callback(x)
        if verbose: 
            print("I=%f radius %f tikonow%f"%(I,radius,tikonow))       
        duration=time.clock()-start
        curve.append(I)
        times.append(duration) 
        if duration>max_duration:
            break  
        
        while 1:
            H2=H+tikonow*scipy.sparse.eye(len(x))
            #deltaX=-scipy.sparse.linalg.spsolve(H2,G)
            deltaX=-scipy.sparse.linalg.bicg(H2,G,maxiter=maxiterCG)[0]

            if np.max(np.abs(deltaX))>radius:
                tikonow=tikonow*2
            else:
                tikonow=tikonow*0.5
                break

        nx=x+deltaX
        if not bounds is None:
            nx=np.maximum(bounds[:,0],nx)
            nx=np.minimum(bounds[:,1],nx)
            
        predictedDeltaEnergyQuadratic=0.5*deltaX.dot(np.array(H*deltaX[:,None]).T.flatten()) + G.dot(deltaX)
        predictedDeltaEnergyLinear= G.dot(deltaX)
        if np.max(np.abs(deltaX))<=tol:
            break

        nI = funcNoJac(nx)
        actualDeltaEnergy=nI-I
        ratioQuadra=actualDeltaEnergy/predictedDeltaEnergyQuadratic
        ratioLinear=actualDeltaEnergy/predictedDeltaEnergyLinear

        if ratioQuadra<0.3 or actualDeltaEnergy>0 or predictedDeltaEnergyLinear>0:
            radius=radius*0.6
        elif ratioQuadra<0.7:
            radius=radius*0.9
        else:
            radius=min(maxradius,radius*1.5)
        if actualDeltaEnergy<0:
            x=nx
    return {'x':x,'curve':curve,'times':times}

def simpleGradientDescent(x0,funcNoJac,funcWithJac,max_duration,max_iter,verbose=True,callback=None):    
    curve=[]
    times=[]  
    start=time.clock()
    step=1
    x=x0.copy()
    for iter in range(max_iter):
        I,G = funcWithJac(x)
        if not callback is None:
            callback(x)        
        duration=time.clock()-start
        curve.append(I)
        times.append(duration) 
        if duration>max_duration:
            break 
        deltaX=-step*G/np.max(np.abs(G))
        nx=x+deltaX
        nI = funcNoJac(nx)
        actualDeltaEnergy=nI-I
        predictedDeltaEnergyLinear= G.dot(deltaX)
        ratioLinear=actualDeltaEnergy/predictedDeltaEnergyLinear
        if ratioLinear<0.2:            
            step=step*0.3
            #print(step)
        elif ratioLinear>0.8:            
            step=step*1.2
            #print(step)
        if nI<I:
            x=nx
            
            
    return {'x':x,'curve':curve,'times':times}

class optimTimer():
    """this class can help in order to keep track of the visite solution duting iterations and to comoute convergence curves"""
    def __init__(self,xopti,evalFunc,ax_curve,ax_dist):
        self.listx=[]
        self.times=[]
        self.start=time.clock()
        self.xopti=xopti
        self.evalFunc=evalFunc
        self.ax_curve=ax_curve
        self.ax_dist=ax_dist
    def clear(self):
        self.start=time.clock()
        self.listx=[]
        self.times=[]        
    def callback(self,x):
        self.listx.append(x.copy())
        self.times.append(time.clock()-self.start)  

    def setOptim(self,xopti):
        self.xopti=xopti
    def stats(self,label):
        distsol=[np.mean((x-self.xopti)**2) for x in self.listx]
        self.ax_dist.semilogy(self.times,distsol,label=label)
        curve=[self.evalFunc(x) for x in self.listx]
        self.ax_curve.plot(self.times,np.minimum.accumulate(curve),label=label)
        self.ax_curve.set_ylabel('energy')
        self.ax_dist.set_ylabel('mean square error to solution')
        self.ax_curve.set_xlabel('duration in sec')
        self.ax_dist.set_xlabel('duration in sec')        
        self.ax_dist.legend()


        
      