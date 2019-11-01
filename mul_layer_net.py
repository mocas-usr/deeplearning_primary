# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:10:08 2019

@author: HAX
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np
from collections import OrderedDict
from errorback import sigmoid,Relu
from affine_softmax import affine,softmaxwithloss
from twolayernet import numerical_gradient

class multi_layer_net:
    def __init__(self,input_size,hidden_layer_list,output_size,activation='relu', weight_init_std='relu', weight_decay_lambda=0):
        self.input_size=input_size
        self.output_size=output_size
        self.hidden_layer_list=hidden_layer_list
        self.hidden_layer_num=len(hidden_layer_list)
        self.activation=activation
        self.weight_init_std=weight_init_std
        self.weight_decay_lambda = weight_decay_lambda
        self.params ={}
        
        ##初始化权重
        self.__init_weight(weight_init_std)
        
        ##生成层
        activation_layer={'sigmoid':sigmoid,'relu':Relu}
        self.layers=OrderedDict()
        for idx in range(1,self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)]=affine(self.params['w'+str(idx)],self.params['b'+str(idx)])
            self.layers['activation_function'+str(idx)]=activation_layer[activation]()
             
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = affine(self.params['w' + str(idx)],self.params['b' + str(idx)])
        self.last_layer = softmaxwithloss()
    
    ##初始权重值
    def __init_weight(self, weight_init_std):
        all_size_list = [self.input_size] + self.hidden_layer_list + [self.output_size]
        
        ##relu函数时的He初始值，sigmoid函数时使用Xavier初始值
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值

            self.params['w' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
     
        
        #预测##
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x
    def loss(self,x,t):
        y=self.predict(x)
        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            w = self.params['w' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(w ** 2)
        return self.last_layer.forward(y, t) + weight_decay
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy 
    def numerical_gradient(self,x,t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['w' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])

        return grads
    def gradient(self,x,t):
        # forward
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
        for idx in range(1, self.hidden_layer_num+2):
            grads['w' + str(idx)] = self.layers['Affine' + str(idx)].dw + self.weight_decay_lambda * self.layers['Affine' + str(idx)].w
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

        return grads
               
            
            
    