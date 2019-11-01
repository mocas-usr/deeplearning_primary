# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:29:26 2019

@author: HAX
"""
#
# ##定义一个与门##
#def AND (x1,x2):
#    w1,w2,theta=0.5,0.5,0.7
#    tmp=w1*x1+w2*x2
#    if tmp<=theta:
#        return 0
#    if tmp>theta:
#        return 1
#A=AND(0,0)
#B=AND(0,1)



###用numpy来构造各种门##  
import numpy as np
def NAND(X1,X2):
    X=np.array([X1,X2])
    W=np.array([-0.5,-0.5])
    B=0.7
    tmp=np.sum(W,X)+B
    if tmp<=0:
        return 0
    else:
        return 1
def OR(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.2
    tmp=np.sun(w,x)+b
    if tmp<=0:
        return 0
    else:
        return 1
def AND(x1,x2):
    x=np.array([x1,x2])
    w=np.array([0.5,0.5])
    b=-0.7
    tmp=np.sum(w,x)+b
    if tmp<=0:
        return 0
    else:
        return 1
##异或门##
def XOR(x1,x2):
#    x=np.array([x1,x2])
#    w=np.array([])
    s1=NAND(x1,x2)
    S2=OR(x1,x2)
    y=AND(s1,S2)
    return y
    

    
    