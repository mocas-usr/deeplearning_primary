# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:19:21 2019

@author: HAX
"""

import numpy as np
from affine_softmax import affine,BatchNormalization,Dropout,softmaxwithloss
from errorback import sigmoid,Relu
from collections import OrderedDict
from twolayernet import numerical_gradient

class multi_net_extend:
    def __init__(self,input_size,hidden_size_list,output_size,
                 activation='relu', weight_init_std='relu', weight_decay_lambda=0,
                 use_dropout = False, dropout_ration = 0.5, use_batchnorm=False):
        ##初始化参数
        self.input_size=input_size
        self.hidden_size_list=hidden_size_list
        self.output_size=output_size
        self.hidden_layer_num = len(hidden_size_list)
        ##初始化网络设置
        self.use_dropout = use_dropout
        self.weight_decay_lambda = weight_decay_lambda
        self.use_batchnorm = use_batchnorm
        self.params = {}
        
        #初始化w权重
        self.__init_weight(weight_init_std)
        
        #生成层，利用affine生成网络结构
        activation_layer={'sigmoid':sigmoid,'relu':Relu}
        self.layers = OrderedDict()
        for idx in range(1,self.hidden_layer_num+1):
            self.layers['Affine'+str(idx)]=affine(self.params['w'+str(idx)],self.params['b'+str(idx)]) 
            
            if self.use_batchnorm:
                self.params['gamma' + str(idx)] = np.ones(hidden_size_list[idx-1])
                self.params['beta' + str(idx)] = np.zeros(hidden_size_list[idx-1])
                self.layers['BatchNorm' + str(idx)] = BatchNormalization(self.params['gamma' + str(idx)], self.params['beta' + str(idx)])
                
            self.layers['Activation_function' + str(idx)] = activation_layer[activation]()
            
            if self.use_dropout:
                self.layers['Dropout' + str(idx)] = Dropout(dropout_ration)
           
            ##最后一层输出层
        idx = self.hidden_layer_num + 1
        self.layers['Affine' + str(idx)] = affine(self.params['w' + str(idx)], self.params['b' + str(idx)])

        self.last_layer = softmaxwithloss()
    def __init_weight(self, weight_init_std):
         all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
         
         ##初始值设定
         for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # 使用ReLU的情况下推荐的初始值
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # 使用sigmoid的情况下推荐的初始值
            self.params['w' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
            
    def predict(self, x, train_flg=False):
        for key, layer in self.layers.items():
            if "Dropout" in key or "BatchNorm" in key:
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)

        return x
    
    def loss(self, x, t, train_flg=False):
        """求损失函数
        参数x是输入数据，t是教师标签
        """
        y = self.predict(x, train_flg)

        weight_decay = 0
        for idx in range(1, self.hidden_layer_num + 2):
            W = self.params['w' + str(idx)]
            weight_decay += 0.5 * self.weight_decay_lambda * np.sum(W**2)

        return self.last_layer.forward(y, t) + weight_decay
    
    def accuracy(self, X, T):
        Y = self.predict(X, train_flg=False)
        Y = np.argmax(Y, axis=1)
        if T.ndim != 1 : T = np.argmax(T, axis=1)

        accuracy = np.sum(Y == T) / float(X.shape[0])
        return accuracy
    
    
    def numerical_gradient(self, X, T):
        """求梯度（数值微分）

        Parameters
        ----------
        X : 输入数据
        T : 教师标签

        Returns
        -------
        具有各层的梯度的字典变量
            grads['W1']、grads['W2']、...是各层的权重
            grads['b1']、grads['b2']、...是各层的偏置
        """
        loss_W = lambda W: self.loss(X, T, train_flg=True)

        grads = {}
        for idx in range(1, self.hidden_layer_num+2):
            grads['w' + str(idx)] = numerical_gradient(loss_W, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_W, self.params['b' + str(idx)])
            
            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = numerical_gradient(loss_W, self.params['gamma' + str(idx)])
                grads['beta' + str(idx)] = numerical_gradient(loss_W, self.params['beta' + str(idx)])

        return grads
        
    def gradient(self, x, t):
        # forward
        self.loss(x, t, train_flg=True)

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
            grads['w' + str(idx)] = self.layers['Affine' + str(idx)].dw + self.weight_decay_lambda * self.params['w' + str(idx)]
            grads['b' + str(idx)] = self.layers['Affine' + str(idx)].db

            if self.use_batchnorm and idx != self.hidden_layer_num+1:
                grads['gamma' + str(idx)] = self.layers['BatchNorm' + str(idx)].dgamma
                grads['beta' + str(idx)] = self.layers['BatchNorm' + str(idx)].dbeta

        return grads