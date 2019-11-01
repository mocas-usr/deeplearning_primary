# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 15:37:26 2019

@author: HAX
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np


##sigmoid函数##
def sigmoid(x):
    return 1/(1+np.exp(-x))
#softmax函数##
def softmax(x):
     if x.ndim == 2:
            x = x.T
            x = x - np.max(x, axis=0)
            y = np.exp(x) / np.sum(np.exp(x), axis=0)
            return y.T 
     x = x - np.max(x) # 溢出对策
     return np.exp(x) / np.sum(np.exp(x))
    
##损失函数##
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

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)   
    
    
##定义两层网络##
class two_layer_net():
    def __init__(self,input_size,hidden_size,output_size,weight_init_std=0.01):
        self.params={}
        self.params['w1']=weight_init_std*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['w2']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
    def predict(self,x):
        w1,w2=self.params['w1'],self.params['w2']
        b1,b2=self.params['b1'],self.params['b2']
        a1=np.dot(x,w1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,w2)+b2
        z2=softmax(a2)
        return z2
    def loss(self,x,t):
        y=self.predict(x)
        error_loss=cross_entropy_error(y,t)
        return error_loss
    def accuracy(self,x,t):
        y=self.predict(x)
        y=np.argmax(y,axis=1)
        t=np.argmax(t,axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy
    def numerical_gradient(self,x,t):
#        def f(w):
#            return self.loss(x,t)
        ##lamda表达式##
        f=lambda w:self.loss(x,t)
        grads={}
        grads['w1']=numerical_gradient(f,self.params['w1'])
        grads['w2']=numerical_gradient(f,self.params['w2'])
        grads['b1']=numerical_gradient(f,self.params['b1'])
        grads['b2']=numerical_gradient(f,self.params['b2'])
        return grads
    def gradient(self, x, t):
        w1, w2 = self.params['w1'], self.params['w2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, w1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, w2) + b2
        y = softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['w2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, w2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads
    
    
##程序运行#
if __name__=='__main__':
    net=two_layer_net(input_size=784,hidden_size=100,output_size=10)
    print('net.w1.shape',net.params['w1'].shape)
    print('w2.shape',net.params['w2'].shape)
    print('b2.shape',net.params['b2'].shape)
    x=np.random.rand(100,784)
    #y=net.predict(x)
    #print('y',y)
    t=np.random.rand(100,10)
    #print(t)
    z2=net.predict(x)
    print('z2',z2)
    loss=net.loss(x,t)
    print('loss',loss)
    grad=net.numerical_gradient(x,t)
    print('grad',grad)


            
        
        
        
        
        