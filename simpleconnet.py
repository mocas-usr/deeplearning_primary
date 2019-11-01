# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 19:21:53 2019

@author: HAX
"""
import numpy as np 
from collections import OrderedDict
from convolutin import convolution,pooling
from errorback import Relu
from affine_softmax import affine,softmaxwithloss
import pickle

class simpleconnet:
    def __init__(self,input_dim=(1,28,28),conv_param={'filter_num':30,'filter_size':5,'pad':0,'stride':1},
                 hidden_size=100,output_size=10,weight_init_std=0.01):
        filter_num=conv_param['filter_num']
        filter_size=conv_param['filter_size']
        filter_pad=conv_param['pad']
        filter_stride=conv_param['stride']
        input_size=input_dim[1]
        conv_output_size=(input_size-filter_size+2*filter_pad)/filter_stride+1
        pool_output_size=int(filter_num*(conv_output_size/2)*(conv_output_size/2))
        

        ##初始化权重
        self.params={}
        self.params['w1']=weight_init_std*np.random.randn(filter_num,input_dim[0],filter_size,filter_size)
        self.params['b1']=np.zeros(filter_num)
        self.params['w2']=weight_init_std*np.random.randn(pool_output_size,hidden_size)
        self.params['b2']=np.zeros(hidden_size)
        self.params['w3']=weight_init_std*np.random.randn(hidden_size,output_size)
        self.params['b3']=np.zeros(output_size)
        
        ##生成层
        self.layers=OrderedDict()
        self.layers['conv1']=convolution(self.params['w1'],self.params['b1'],
                   conv_param['stride'],conv_param['pad'])
        self.layers['relu1']=Relu()
        self.layers['pool1']=pooling(pool_h=2,pool_w=2,stride=2)
        self.layers['affine1']=affine(self.params['w2'],self.params['b2'])
        self.layers['relu2']=Relu()
        self.layers['affine2']=affine(self.params['w3'],self.params['b3'])
        self.last_layer=softmaxwithloss()
    def predict(self,x):
        for layer in self.layers.values():
            x=layer.forward(x)
        return x
    def loss(self,x,t):
        y=self.predict(x)
        return self.last_layer.forward(y,t)
    
    
#        ##forward
#        self.loss(x,t)
#        
#        dout=1
#        dout=self.last_layer.backward(dout)
#        
#        layers=list(self.layers.values())
#        layers.reverse()
#        for layer in layers:
#            dout=layer.backward(dout)
#            grads={}
#            grads['w1']=self.layers['conv1'].dw
#            grads['b1']=self.layers['conv2'].db
#            grads['w2']=self.layers['affine1'].dw
#            grads['b2']=self.layers['affine1'].db
#            grads['w3']=self.layers['affine2'].dw
#            grads['b3']=self.layes['affine2'].db
#            return grads
        
    def gradient(self,x,t):
        
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 设定
        grads = {}
        grads['w1'], grads['b1'] = self.layers['conv1'].dw, self.layers['conv1'].db
        grads['w2'], grads['b2'] = self.layers['affine1'].dw, self.layers['affine1'].db
        grads['w3'], grads['b3'] = self.layers['affine2'].dw, self.layers['affine2'].db

        return grads
    
    def accuracy(self, x, t, batch_size=100):
        
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]
       
    def save_params(self,file_name='params.pkl'):
        params={}
        for key,val in self.params.items():
            params[key]=val
        
        with open(file_name,'wb') as f:
            pickle.dump(params,f)
    
    
    def load_params(self, file_name="params.pkl"):
        with open(file_name,'rb') as f:
            params = pickle.load(f)
        for key,val in self.params.items():
            params[key]=val
        for i, key in enumerate(['conv1', 'affine1', 'affine2']):
            self.layers[key].w = self.params['w' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
        
        
        
        