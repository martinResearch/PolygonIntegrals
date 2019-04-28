import numpy as np
from polygonintegrals.antialiasedPolygon import drawAntialiasedPolygon
import matplotlib.pylab as plt

ims=None
for t in np.linspace(0,20,200):
    im=np.zeros((60,60,3))    
    poly=np.array([[10,10],[50,31.5],[14,50-t]])
    drawAntialiasedPolygon(im,poly, [1,0,0], 'bilinear')
    poly=np.array([[30,50],[20,31],[42,10+t]])
    drawAntialiasedPolygon(im,poly, [0.3,0.3,1], 'bilinear')
    if ims is None:
        plt.ion()
        ims=plt.imshow(im,interpolation='none')
        plt.draw()
        plt.show()        
        
    else:
        ims.set_array(im)
        plt.draw()



