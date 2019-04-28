from activePolygon import *
from ad.admath import atan2
import ad  #https://pypi.python.org/pypi/ad/1.2.3
import  algopy
from algopy import UTPM, exp,reshape

# this file is use to make some test with a simple rigid prior for active polygons
# we are trying to use some python utomatic differentiztion tools to get the jcobian and hessian of the rigidity score
# - ad seems slow 
# - algopy seems not mature enough yet to handle well numpy expression

class ARAPPolygonPrior():
    '''as rigid as possible prior that penalizes change in lenghts of the polygon edges and change in the angles at the corners'''
    def __init__(self,reference_shape,coef_lenghts,coef_angles):
        self.reference_shape=reference_shape
        self.reference_lengths,self.reference_angles=self.getEdgesLenghtsAndCornerAngles(reference_shape)
        self.coef_lenghts=coef_lenghts
        self.coef_angles=coef_angles
        
    def getEdgesLenghtsAndCornerAngles(self,shape):
        #compute lenghts and angles
        diff=shape-np.roll(shape,1,axis=0)
        lengths=np.sqrt(np.sum(diff**2,axis=1))
        directions=diff/lengths[:,None]
        rot=np.array([[0,1],[-1,0]])
        x=np.sum(directions*np.roll(directions,1,axis=0),axis=1)
        y=np.sum(directions*np.roll(directions,1,axis=0).dot(rot),axis=1)
        if type(x[0])==numpy.float64:
            angles=np.arctan2(x,y)
        else:# i ma not able to test if it is an ad number
            angles=np.array(atan2(x,y))# for automatic differentiation to work
        
            
        
         
        
        return lengths,angles        
        
    def eval(self,shape):        
        lengths,angles=self.getEdgesLenghtsAndCornerAngles(shape.reshape((-1,2)))
        cost_lenght=self.coef_lenghts*np.sum((lengths-self.reference_lengths)**2)
        cost_angles=self.coef_angles*np.sum((angles-self.reference_angles)**2)        
        return cost_lenght+cost_angles
    
if __name__ == '__main__':
    plt.ion()
        
    # get a random polygon
    shape1=np.random.rand(4,2)*100
    polyPrior=ARAPPolygonPrior(shape1,1,1)
    
    
    # perturb the polygon and minimize the rigid cost plus a term that penalizes devition from the pertubed polyon
    shape2=shape1+np.random.randn(4,2)*10
    plotClosedPolygon(shape2)
    
    plotClosedPolygon(shape1)
    print(polyPrior.eval(shape2))
    
    
    #grad, hess = ad.gh(polyPrior.eval)
    
    
    #res=scipy.optimize.minimize(polyPrior.eval, shape2.flatten().copy(),jac=grad, method='CG') 
    #shape3=res['x'].reshape(-1,2)
    #print(polyPrior.eval(shape3))
    #plotClosedPolygon(shape3)
    #plt.show()
    #print('done')
    
    
    #x= UTPM.init_jacobian(shape2.flatten())
    #y = polyPrior.eval(x)
    #algopy_jacobian = UTPM.extract_jacobian(y)
    #print('jacobian = ',algopy_jacobian)    
    
    cg = algopy.CGraph()
    x = algopy.Function(shape2.flatten())
    y = polyPrior.eval(x)
    cg.trace_off()
    cg.independentFunctionList = [x]
    cg.dependentFunctionList = [y]  
    print('gradient =', cg.gradient([3.,5,7]))
    






        
        
