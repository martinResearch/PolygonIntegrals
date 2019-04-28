from polygonintegrals.integreWithinPolygon import Image
from polygonintegrals.xorshift import xorshift
import pstats, cProfile
import time

def _test2(display=False):   
    
    

    rd=xorshift()      
   
    Im = Image(rd.random(300, 300))
    N = 1000
    P = rd.random(2, N).T*296.+2.# TO DO : if we get closer tho the borders of the image it seems like the gradient becomes false
    
    #precompute cumulative sum
    Im.cumsum0()
    
    start = time.clock()
    Ic = Im.integreWithinPolygon(P, 'const',nargout=1)
    elapsed = (time.clock() - start)
    print('const nargout=1: ' + str(elapsed))   
    
    start = time.clock()
    Ic,dIc_dP = Im.integreWithinPolygon(P, 'const',nargout=2)
    elapsed = (time.clock() - start)
    print('const nargout=2: ' + str(elapsed))     
    
    start = time.clock()
    Ic, dIc_dP, Hessianc,= Im.integreWithinPolygon(P, 'const',nargout=3)
    elapsed = (time.clock() - start)
    print('const nargout=3: '  + str(elapsed))
    
    start = time.clock()
    Ic, dIc_dP, Hessianc, Mc = Im.integreWithinPolygon(P, 'const',nargout=4)
    elapsed = (time.clock() - start)
    print('const nargout=4: '  + str(elapsed))
    

    start = time.clock()
    Ic = Im.integreWithinPolygon(P, 'bilinear',nargout=1)
    elapsed = (time.clock() - start)
    print('bilinear nargout=1: ' + str(elapsed))    
    
    start = time.clock()
    Ic,dIc_dP = Im.integreWithinPolygon(P, 'bilinear',nargout=2)
    elapsed = (time.clock() - start)
    print('bilinear  nargout=2: ' + str(elapsed))   
    
    start = time.clock()
    Ib, dIb_dP, Hessianb, Mb = Im.integreWithinPolygon(P, 'bilinear',nargout=4)
    elapsed = (time.clock() - start)
    print('bilinear nargout=4: '  + str(elapsed))        
    
    #if display:
        #plt.figure
        #plt.imshow(Mc.conj().T,cmap = plt.get_cmap('gray'))
        #plt.show()
    
        #plt.figure
        #plt.imshow(Mb.conj().T,cmap = plt.get_cmap('gray'))
        #plt.show()    
        
    assert abs( Mc.flatten().dot(Im.values.flatten())[0,0]- Ic)<1e-10
    assert abs( Mb.flatten().dot(Im.values.flatten())[0,0]-Ib)<1e-10
    
    
    
 
    
   

if __name__ == '__main__':
    
    #_test2(True)
    
    cProfile.runctx("_test2(True)", globals(), locals(), "Profile.prof")
    
    s = pstats.Stats("Profile.prof")
    s.strip_dirs().sort_stats("time").print_stats()    
    

