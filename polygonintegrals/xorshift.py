#cython: profile=True
#cython: boundscheck=False    
#cython: wraparound=False
#cython: initializedcheck=False
#cython: nonecheck=False
#cython: cdivision=True
#cython: overflowcheck=False
#cython: overflowcheck.fold=False

import numpy

#cython#cimport libc.math  as math
#cython#cimport numpy as np
#cython#cimport cython

#cython#cdef unsigned long Oxffffffff
Oxffffffff =0xffffffff 



#cython#cdef\
class xorshift:
        # simple pseudo-random generator base on the xorshift algorithm
        # the goal is to get a random generator that can easily be reimpleted in various languages(matlab for example) in order to be able to generate
        # the same random number sequences in various languages
        # http://stackoverflow.com/questions/5829499/how-to-let-boostrandom-and-matlab-produce-the-same-random-numbers
        # http://stackoverflow.com/questions/3722138/is-it-possible-to-reproduce-randn-of-matlab-with-numpy        
        
        #cython#cdef unsigned long x,y,z,w
        #cython#cdef float max
        
        def __init__(self):
                
                self.x = 123456789
                self.y = 362436069
                self.z = 521288629
                self.w = 88675123
                self.max=2**32
        
        
        def next(self):#cython_delete_line#
        #cython#cdef unsigned long next(self):
                #cython#cdef unsigned long t
                #cython#cdef unsigned long w
                t = self.x ^ (self.x<<11)& Oxffffffff                   # <-- keep 32 bits
                
                self.x = self.y
                self.y = self.z
                self.z = self.w
                w = self.w
                self.w = w ^ (w >> 19) ^(t ^ (t >> 8))& Oxffffffff                   # <-- keep 32 bits
                return self.w
        def random(self,\
                   #cython#int\
                   m=1,\
                   #cython#int\
                   n=1):
                #cython#cdef np.ndarray[np.float64_t, ndim=2]\
                r=numpy.empty((m,n))               
                #cython#cdef unsigned int i
                #cython#cdef unsigned int j
                for i in range(m):
                        for j in range(n):
                                r[i,j]=float(self.next())/self.max
                return r
            
        def randint(self,a,b):            
                r=int(a+(b-a+1)*self.random())
                return r
        
        
        def choice( self,set):
                i=self.randint(0,len(set)-1)
                return set[i]
            
        
        def gauss(self,mean,std):
                # generate nromal distributed pseudo random numbers using the box-muller transform
                # http://en.wikipedia.org/wiki/Box-Muller_transform
                u1=self.random()
                u2=self.random()
                return mean+std*math.sqrt(-2*math.log(u1))*math.cos(2*math.pi*u2)
