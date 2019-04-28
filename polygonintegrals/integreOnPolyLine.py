
import numpy as np
import scipy


# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def integreOnPolyLine(P, f, varargin):


    #% integreOnPolygonBoundary(P,f,F)
    #% Integre an image f along the boundary of a polygon
    #% Assume f is interpolated using bilinear interpolation
    #%
    #% INPUT :
    #% P : vertices of the polygon , matrix of sier 2 by N
    #% f : the discretized 2D function beeing integrated
    #%
    #% OUTPUT
    #% I     : the integral
    #% dI_dP : the derivative dI/dP
    #% M : the antialiased boundary of the polygon
    #% (intensity of pixel=lenght of the subpixel fragment)
    #%
    #% coded by Martin de la Gorce
    #%
    #% EXAMPLE
    #%
    #% f=rand(100,100);
    #% N=3
    #% P=rand(2,N)*90+5
    #% [Ic,dIc_dP]=integreOnPolyLine(P,f,'bilinear',1);
    #% [Ib,dIb_dP,HessianI]=integreOnPolyLine(P,f,'bicubic',1);
    #% [Ic,dIc_dP,HessianI,Mc]=integreOnPolyLine(P,f,'const');
    #% [Ib,dIb_dP,HessianI,Mb]=integreOnPolyLine(P,f,'bilinear');
    #% 
    #% figure;imshow(Mc',[]);axis xy;
    #% figure;imshow(Mb',[]);axis xy;
    #% Ic,sum(Mc(:).*f(:))
    #% Ib,sum(Mb(:).*f(:))
    #% func=@(x) integreOnPolyLine(x,f,'bilinear');
    #% verif_gradient(func,P)
    #% func=@(x) integreOnPolyLine(x,f,'bilinear',1);
    #% verif_gradient(func,P)
    #% func=@(x) integreOnPolyLine(x,f,'bicubic',1);
    #% verif_gradient(func,P)
    E = np.array(np.vstack((np.hstack((np.arange(1., (P.shape[1]-1.)+1))), np.hstack((np.arange(2., (P.shape[1])+1))))))
    if nargout == 1.:
        I = integreOnSegments(P, E, f, varargin.cell[:])
    elif nargout == 2.:
        I, dI_dP = integreOnSegments(P, E, f, varargin.cell[:])
        
    elif nargout == 3.:
        I, dI_dP, HessianI = integreOnSegments(P, E, f, varargin.cell[:])
        
    elif nargout == 4.:
        I, dI_dP, HessianI, M = integreOnSegments(P, E, f, varargin.cell[:])
        
    
    return I, dI_dP, HessianI, M
