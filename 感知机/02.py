# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 19:41:33 2019

@author: HAX
"""

#def step_function(x):
#    if x>0:
#        return 1
#    else:
#        return 0

##几个基本函数##
import numpy as np
def step_function(x):
    y=x>0
    return y.astype(np.int)
def sigmoid(x):
    y=1/(1+np.exp(-x))
    return y
def relu(x):
    return np.maximum(x,0)


##维度对应##
A=np.array([[1,2],[3,4],[5,6]])
print(A.shape)
B=np.array([[1,4,5],[6,8,4]])
print(B.shape)
sumab=np.dot(A,B)
sumba=np.dot(B,A)
##三层网络的实现##
x1=np.array([5,6,8])
print(x1.shape)
w1=np.array([[1,5,6],[5,8,9],[1,4,6]])
print(w1.shape)
b1=np.array([0.5,0.5,0.58])
a1=np.dot(x1,w1)+b1
z1=sigmoid(a1)
print("a1=",a1)
print("z1=",z1)
#一层网络搭建结束#
w2=np.array([[5,8],[1,6],[4,7]])
b2=np.array([0.5,0.6])
a2=np.dot(z1,w2)+b2
z2=sigmoid(a2)
##两层网络搭建结束##
w3=np.array([[2,4],[8,6]])
b3=np.array([0.1,0.5])
a3=np.dot(z2,w3)+b3
z3=sigmoid(a3)
print('z3',z3)
##三层网络搭建结束##

##另一种搭建网络的代码##
def init_network():
    network={}
    network['w1']=np.array([[1,5,6],[5,8,9],[1,4,6]])
    network['b1']=np.array([0.5,0.5,0.58])
    network['w2']=np.array([[5,8],[1,6],[4,7]])
    network['b2']=np.array([0.5,0.6])
    network['w3']=np.array([[2,4],[8,6]])
    network['b3']=np.array([0.1,0.5])
    return network
def forward(network,x):
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    z3=sigmoid(a3)
    return z3

##主程序##
network=init_network()
x1=np.array([5,6,8])
z3=forward(network,x1)
print('第二个z3,',z3)


##softmax函数##
##因为exp函数之后数值过大，会出现溢出问题##
#def softmax(x):
#    exp_x=np.exp(x)
#    sum_exp_x=np.sum(exp_x)
#    y=exp_x/sum_exp_x
#    return y
def softmax(x):
    c=np.max(x)
    exp_xc=np.exp(x-c)
    sum_exp_xc=np.sum(exp_xc)
    y=exp_xc/sum_exp_xc
    return y



    

