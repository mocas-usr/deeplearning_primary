# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 20:43:08 2019

@author: HAX
"""

import numpy as np
from backwardnet import backnet

##定义获取minibatch的函数
def get_batch(x,t,train_size,batch_size):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x[batch_mask]
    t_batch=t[batch_mask]
    return x_batch,t_batch

##SGD优化
class SGD:
    def __init__(self,lr=0.01):
        self.lr=lr
    def update(self,params,grads):
        for key in params.keys():
            params[key]-=self.lr*grads[key]

##momentum优化
class momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr=lr
        self.momentum=momentum
        self.v=None
        
    def update(self,params,grads):
        if self.v is None:
            self.v={}
            for key,val in params.items():
                self.v[key]=np.zeros_like(val)
        for key in params.keys():
            self.v[key]=self.momentum*self.v[key]-self.lr*grads[key]
            params[key]+=self.v[key]
            
##adagrad
class adagrad:
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None
    def update(self,params,grads):
        if self.h==None:
            self.h={}
            for key,val in params.items():
                self.h[key]=np.zeros_like(val)
        for key in params.keys():
            self.h[key]+=grads[key]*grads[key]
            params[key]-=self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7)

class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            #self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            #self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            
            #unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            #unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            #params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

            
            
if __name__=='__main__':
    network=backnet(input_size=784,hidden_size=50,output_size=10)
    optimizer=SGD()
    x=np.random.rand(100,784)
    t=np.random.rand(100,10)
    
    for i in range(1000):
        x_batch,t_batch=get_batch(x,t,train_size=100,batch_size=10)
        grads=network.gradiet(x_batch,t_batch)
        parms=network.params
        optimizer.update(parms,grads)
        if i%200==0:
            print(parms)
