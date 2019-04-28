
import numpy as np
import scipy
from integreOnSegments import integreOnSegments


try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def integreOnPolygonBoundary(P, f, interp,nargout):

   
   
    # integreOnPolygonBoundary(P,f,F)
    # Integre an image f along the boundary of a polygon
    # Assume f is interpolated using bilinear interpolation
    #
    # INPUT :
    # P : vertices of the polygon , matrix of sier 2 by N
    # f : the discretized 2D function beeing integrated
    #
    # OUTPUT
    # I     : the integral
    # dI_dP : the derivative dI/dP
    # M : the antialiased boundary of the polygon
    # (intensity of pixel=lenght of the subpixel fragment)
    #
    # coded by Martin de la Gorce
    #
    # EXAMPLE
    #
    # f=rand(100,100);
    # N=3
    # P=rand(2,N)*100
    # [Ic,dIc_dP,Mc]=integreOnPolygonBoundary(P,f,'const');
    # [Ib,dIb_dP,Mb]=integreOnPolygonBoundary(P,f,'bilinear');
    # figure;imshow(Mc',[]);axis xy;
    # figure;imshow(Mb',[]);axis xy;
    # Ic,sum(Mc(:).*f(:))
    # Ib,sum(Mb(:).*f(:))
    # func=@(x) integreOnPolygonBoundary(x,f,'bilinear');
    # verif_gradient(func,P)

    if P.shape[1] != 2 or P.shape[0]<=2:
        raise 'P sould be of size 2 by N with N>=2'
	
    
    
    E = np.array(np.vstack((np.hstack((np.arange(1, (P.shape[1])+1))), np.hstack((np.array(np.hstack((np.arange(2, (P.shape[1])+1), 1))))))))
    if nargout == 1:
        I = integreOnSegments(P, E, f, interp,nargout)
        return I
    elif nargout == 2:
        I, dI_dP = integreOnSegments(P, E, f, interp,nargout)
        return I, dI_dP
    elif nargout == 3:
        I, dI_dP, HessianI = integreOnSegments(P, E, f, interp,nargout)
        return I, dI_dP, HessianI
    elif nargout == 4:
        I, dI_dP, HessianI, M = integreOnSegments(P, E, f, interp,nargout)
        return I, dI_dP, HessianI, M
