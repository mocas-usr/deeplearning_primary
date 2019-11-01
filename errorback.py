# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 18:57:04 2019

@author: HAX
"""

import numpy as np

##实现乘法层##
class mullayer:
    def __init__(self):
        self.x=None
        self.y=None
        
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y
        return out
    def backward(self,dout):
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy
    
##简单实现正向传播##
apple=100
apple_num=2
tax=1.1

#layer
mul_apple_layer=mullayer()
mul_tax_layer=mullayer()

apple_price=mul_apple_layer.forward(apple,apple_num)
price=mul_tax_layer.forward(apple_price,tax)

print(price)

#backward#
dprice=1
dapple_price,dtax=mul_tax_layer.backward(dprice)
dapple,dapple_num=mul_apple_layer.backward(dapple_price)
print(dapple,dapple_num,dtax)


##实现加法层
class addlayer:
    def __init__(self):
        pass
    def forward(self,x,y):
        out=x+y
        return out
    def backward(self,dout):
        dx=dout*1
        dy=dout*1
        return dx,dy




##实现relu层
class Relu:
    def __init__(self):
        self.mask=None
    def forward(self,x):
        self.mask=(x<=0)
        out=x.copy()
        out[self.mask]=0
        return out
    def backward(self,dout):
        dout[self.mask]=0
        dx=dout
        return dx
x=np.array([[1.0,-0.5],[-2.0,3.0]])
print(x)
mask=(x<=0)
print(mask)
relu=Relu()
x=relu.forward(x)
print('x',x)
dout=x
out=relu.forward(dout)
print('dout',dout)


##sigmoid层实现##
class sigmoid:
    def __init__(self):
        self.out=None
    def forward(self,x):
        out=1/(1+np.exp(-x))
        self.out=out
        return out
    def backward(self,dout):
        dx=dout*(1.0-self.out)*self.out
        return dx
        
        