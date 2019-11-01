# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 20:04:17 2019

@author: HAX
"""
##简单实现过程#
import sys, os
sys.path.append(os.pardir) 
import numpy as np

##softmax函数##
def softmax(x):
    if x.ndim==2:
        x=x.T
        x=x-np.max(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    x=x-np.max(x)
    y=np.exp(x)/np.sum(np.exp(x))
    return y
 
 ##误差函数##   
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)
             
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


##梯度下降函数##
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad

##简单simplenet类##
class simpleNet:
    def __init__(self):
        self.w= np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.w)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss
 
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

#f = lambda w: net.loss(x, t)  ###伪参数是什么意思，有什么作用？
##等效函数##f(m)或者f(w)的作用都是对应nunerical_gradient的f(x),f(x+h)等
def f(m):
    return net.loss(x,t)

dW = numerical_gradient(f, net.w)

print(dW)
