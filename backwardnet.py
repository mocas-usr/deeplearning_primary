# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 20:50:08 2019

@author: HAX
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from affine_softmax import affine,softmaxwithloss
from errorback import Relu
from twolayernet import numerical_gradient
class backnet:
    def __init__(self,input_size,hidden_size,output_size,weight=0.01):
        self.params={}
        self.params['w1']=weight*np.random.randn(input_size,hidden_size)
        self.params['b1']=np.zeros(hidden_size)
        self.params['w2']=np.random.randn(hidden_size,output_size)
        self.params['b2']=np.zeros(output_size)
        
        ##生成层
        self.layers=OrderedDict()
        self.layers['Affine1']=affine(self.params['w1'],self.params['b1'])
        self.layers['Relu1']=Relu()
        self.layers['Affine2']=affine(self.params['w2'],self.params['b2'])
        self.lastlayer=softmaxwithloss()
    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x
    
    def loss(self,x,t):
        y=self.predict(x)
        return self.lastlayer.forward(y,t)
    def accuracy(self,x,t):
#        y=self.predict(x)
#        y=np.argmax(y,axis=1)
#        if y.ndim!=1:t=np.argmax(t,axis=1)
#        accuracy=np.sum(y==t)/float(x.shape[0])
#        return accuracy
         y=self.predict(x)
         y=np.argmax(y,axis=1)
         t=np.argmax(t,axis=1)
         accuracy=np.sum(y==t)/float(x.shape[0])
         return accuracy
    def numerical_gradient(self,x,t):
        loss_w=lambda w:self.loss(x,t)
        grads={}
        grads['w1']=numerical_gradient(loss_w,self.params['w1'])
        grads['b1']=numerical_gradient(loss_w,self.params['b1'])
        grads['w2']=numerical_gradient(loss_w,self.params['w2'])
        grads['b2']=numerical_gradient(loss_w,self.params['b2'])
        return grads
    
    def gradient(self,x,t):
        self.loss(x,t)
        dout=1
        dout=self.lastlayer.backward(dout)
        layers=list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout=layer.backward(dout)
        grads={}
        grads['w1']=self.layers['Affine1'].dw
        grads['b1']=self.layers['Affine1'].db
        grads['w2']=self.layers['Affine2'].dw
        grads['b2']=self.layers['Affine2'].db
        return grads
    
        