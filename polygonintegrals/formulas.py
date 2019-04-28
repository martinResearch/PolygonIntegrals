from sympy import *
x, y, xa, xb, ya, yb, f11, f01, f10, f00=symbols('x y xa xb ya yb f11 f01 f10 f00')

Fx=integrate(f,x)

m=(xa-xb)/(ya-yb)
b=xa-m*ya;
pfx=integrate(f,(x,0,m*y+b))
s=integrate(pfx,y,ya,yb)


          #
          
          # matlab symbolic calculation that were used to find these expressions :
          #
          # syms  x y xa xb ya yb f11 f01 f10 f00
          # f=x*y*f11+x*(1-y)*f10+(1-x)*y*f01+(1-x)*(1-y)*f00
          # m=(xa-xb)/(ya-yb)
          # b=xa-m*ya;
          # pfx=int(f,x,0,m*y+b);
          # s=int(pfx,y,ya,yb)
          # ss=simple(s)
          #
          # xc=(xa+xb)/2;
          # yc=(ya+yb)/2;
          # fc=(1-xc)*(1-yc)*f00+xc*(1-yc)*f10+(1-xc)*yc*f01+xc*yc*f11
          # sa=(yb-ya)*(xa+xb)/2*fc
          # sa2=limit(pfx,y,yc)
          # simple(ss-sa2)
          # syms w h xce yce
          # xa=xce-w/2
          # ya=yce-h/2
          # xb=xce+w/2
          # yb=yce+h/2
          #
          # simple(eval(c00))
          # %c00=1/24*h*(24*xce-24*xce*yce-2*w*h-12*xce^2-w^2+12*yce*xce^2+yce*w^2+2*h*xce*w)
          # simple(eval(c01))
          # %c01=-1/24*h*(-24*xce*yce-2*w*h+12*yce*xce^2+yce*w^2+2*h*xce*w)
          # simple(eval(c11))
          # %c11=1/24*h*(2*h*xce*w+yce*w^2+12*yce*xce^2)
          # simple(eval(c10))
          # %c10=-1/24*h*(2*h*xce*w-w^2+yce*w^2+12*yce*xce^12*xce^2)
          # simple(eval(c10)+eval(c11))
          # %c10+c11=1/24*h*(w^2+12*xce^2)
          # simple(eval(c01)+eval(c11))
          # %c01+c11=h*xce*yce+1/12*w*h^2
          # simple(eval(c01)+eval(c00))
          # % c01+c00=-1/24*h*w^2+xce*h-1/2*h*xce^2
          # simple(eval(c10)+eval(c00))
          # % c10+c00=xce*h-h*xce*yce-1/12*w*h^2
          # simple(eval(c10)-eval(c01))
          #            
          
          
          
          
          
          
          
          
          
          
          
          
              #%
              #% syms t xa xb ya yb f11 f01 f10 f00
              #% x=xa*t+(1-t)*xb
              #% y=ya*t+(1-t)*yb
              #% f=x*y*f11+x*(1-y)*f10+(1-x)*y*f01+(1-x)*(1-y)*f00
              #% s=f,t,0,1)
              #% simple(s)
              #% xc=(xa+xb)/2
              #% yc=(ya+yb)/2
              #% fc=(1-xc)*(1-yc)*f00+xc*(1-yc)*f10+(1-xc)*yc*f01+xc*yc*f11
              #% simple(s-fc)
              #%
              #% 1/3*(-xa+xb)*(ya-yb)*f01+1/3*(xa-xb)*(-ya+yb)*f10+1/3*(xa-xb)*(ya-yb)*f11+1/3*(-xa+xb)*(-ya+yb)*f00+1/2*(xb*(-ya+yb)+(xa-xb)*(-yb+1))*f10+1/2*(xb*(ya-yb)+(xa-xb)*yb)*f11+1/2*((1-xb)*(-ya+yb)+(-xa+xb)*(-yb+1))*f00+1/2*((1-xb)*(ya-yb)+(-xa+xb)*yb)*f01+f11*xb*yb+(1-xb)*(-yb+1)*f00+(1-xb)*yb*f01+xb*(-yb+1)*f10
              #%  c01=1/3*(xb-xa)*(ya-yb)+1/2*((1-xb)*(ya-yb)+(xb-xa)*yb)+(1-xb)*yb;
              #%  c11=1/3*(xa-xb)*(ya-yb)+1/2*(xb*(ya-yb)+(xa-xb)*yb)+xb*yb;
              #%  c10=1/3*(xa-xb)*(yb-ya)+1/2*(xb*(yb-ya)+(xa-xb)*(1-yb))+xb*(1-yb);
              #%  c00=1/3*(xb-xa)*(yb-ya)+1/2*((1-xb)*(yb-ya)+(xb-xa)*(1-yb))+(1-xb)*(1-yb);
          
              #%  s=f00*c00+f10*c10+f01*c01+f11*c11;
              #%
              #% xc=(xa+xb)/2
              #% yc=(ya+yb)/2
              #% simple(c11-xc*yc)
              #% % 1/12*(xa-xb)*(ya-yb)
              #% simple(c10-xc*(1-yc))
              #% %-1/12*(xa-xb)*(ya-yb)
              #% simple(c01-(1-xc)*yc)
              #% % -1/12*(xa-xb)*(ya-yb)
              #% simple(c00-(1-xc)*(1-yc))
              #% %1/12*(xa-xb)*(ya-yb)   
              
              
              
              
              
              
              
              
              
                  # from sympy import Symbol
                  #
              
                  # matlab symbolic calculation that were used to find these expressions :
                  #
                  # syms  x y xa xb ya yb f11 f01 f10 f00
                  # f=x*y*f11+x*(1-y)*f10+(1-x)*y*f01+(1-x)*(1-y)*f00
                  # m=(xa-xb)/(ya-yb)
                  # b=xa-m*ya;
                  # pfx=int(f,x,0,m*y+b);
                  # s=int(pfx,y,ya,yb)
                  # ss=simple(s)
                  #
                  # xc=(xa+xb)/2;
                  # yc=(ya+yb)/2;
                  # fc=(1-xc)*(1-yc)*f00+xc*(1-yc)*f10+(1-xc)*yc*f01+xc*yc*f11
                  # sa=(yb-ya)*(xa+xb)/2*fc
                  # sa2=limit(pfx,y,yc)
                  # simple(ss-sa2)
                  # syms w h xce yce
                  # xa=xce-w/2
                  # ya=yce-h/2
                  # xb=xce+w/2
                  # yb=yce+h/2
                  #
                  # simple(eval(c00))
                  # %c00=1/24*h*(24*xce-24*xce*yce-2*w*h-12*xce^2-w^2+12*yce*xce^2+yce*w^2+2*h*xce*w)
                  # simple(eval(c01))
                  # %c01=-1/24*h*(-24*xce*yce-2*w*h+12*yce*xce^2+yce*w^2+2*h*xce*w)
                  # simple(eval(c11))
                  # %c11=1/24*h*(2*h*xce*w+yce*w^2+12*yce*xce^2)
                  # simple(eval(c10))
                  # %c10=-1/24*h*(2*h*xce*w-w^2+yce*w^2+12*yce*xce^12*xce^2)
                  # simple(eval(c10)+eval(c11))
                  # %c10+c11=1/24*h*(w^2+12*xce^2)
                  # simple(eval(c01)+eval(c11))
                  # %c01+c11=h*xce*yce+1/12*w*h^2
                  # simple(eval(c01)+eval(c00))
                  # % c01+c00=-1/24*h*w^2+xce*h-1/2*h*xce^2
                  # simple(eval(c10)+eval(c00))
                  # % c10+c00=xce*h-h*xce*yce-1/12*w*h^2
                  # simple(eval(c10)-eval(c01))
                  #                