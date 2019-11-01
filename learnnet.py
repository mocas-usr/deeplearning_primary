# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 09:55:04 2019

@author: HAX
"""

import numpy as np
import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

##定义sigmoid函数
def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_grad(x):
    y=1/(1+np.exp(-x))
    return (1-y)*y
#定义softmax函数
def softmax(x):
    if x.ndim==2:
        x=x.T
        x=x-np.max(x,axis=0)
        y=np.exp(x)/np.sum(np.exp(x),axis=0)
        return y.T
    x=x-np.max(x)
    y=np.exp(x)/np.sum(np.exp(x))
    return y
        
##定义交叉熵误差函数#
def cross_entropy_error(y,t):
    if y.ndim==1:
        y=np.reshape(1,y.size)
        t=np.reshape(1,t.size)
    ##将onehot形式转为代标签的形式
    if y.size==t.size:
        t=np.argmax(t,axis=1)
    
    batch_size=y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

#定义梯度下降函数
def numerical_gradient(f,x):
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


class train_net:
    ##初始化参数
    def __init__(self,input_size,hidden_size,output_size):
        self.parms={}
        self.parms['w1']=np.random.rand(784,50)
        self.parms['b1']=np.zeros(50)
        self.parms['w2']=np.random.rand(50,10)
        self.parms['b2']=np.zeros(10)
    ##前向传播
    def predict(self,x):
        w1=self.parms['w1']
        w2=self.parms['w2']
        b1=self.parms['b1']
        b2=self.parms['b2']
        a1=np.dot(x,w1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,w2)+b2
        z2=softmax(a2)
        return z2
    def loss(self,x,t):
        y=self.predict(x)
        loss=cross_entropy_error(y,t)
        return loss
    def numerical_gradient(self,x,t):
        f=lambda w:self.loss(x,t)
        grads={}
        grads['w1']=numerical_gradient(f,self.parms['w1'])
        grads['w2']=numerical_gradient(f,self.parms['w2'])
        grads['b1']=numerical_gradient(f,self.parms['b1'])
        grads['b2']=numerical_gradient(f,self.parms['b2'])
        return grads
    def gradient(self,x,t):
        #参数初始化
        w1=self.parms['w1']
        w2=self.parms['w2']
        b1=self.parms['b1']
        b2=self.parms['b2']
        ##前向传播
        a1=np.dot(x,w1)+b1
        z1=sigmoid(a1)
        a2=np.dot(z1,w2)+b2
        z2=softmax(a2)
        
        #反向传播
        batch_num=x.shape[0]
        grads={}
        dz=(z2-t)/batch_num
        grads['w2']=np.dot(z1.T,dz)
        grads['b2']=np.sum(dz,axis=0)
        
        da1=np.dot(dz, w2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['w1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)
        return grads
    def accuracy(self,x,t):
         y=self.predict(x)
         y=np.argmax(y,axis=1)
         t=np.argmax(t,axis=1)
         accuracy=np.sum(y==t)/float(x.shape[0])
         return accuracy
        
    
##程序运行测试
if __name__=='__main__':
    #定义神经网络,初始输入
#    net=train_net(input_size=784,hidden_size=50,output_size=10)
#    x=np.random.rand(100,784)
#    t=np.random.rand(100,10)
#    loss=net.loss(x,t)
#    grad=net.gradient(x,t)
#    print(loss)
    #print(grad)
    
    #获取数据
    (x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)
    net=train_net(input_size=100,hidden_size=50,output_size=100)
    #初始化超参数
    iter_num=1000
    learning_rate=0.3102
    epoch_num=50
    train_size=x_train.shape[0]
    batch_size=100
    train_loss_list=[]
    
    
    #获取minibatch
    for i  in range(iter_num):
        batch_mask=np.random.choice(train_size,batch_size)
        x_batch=x_train[batch_mask]
        t_batch=t_train[batch_mask]
        
    ##梯度下降,参数更新
        grads=net.gradient(x_batch,t_batch)
        for key in ('w1','w2','b1','b2'):
            net.parms[key]-=learning_rate*grads[key]
    
    ##求损失
        batch_loss=net.loss(x_batch,t_batch)
        train_loss_list.append(batch_loss)
        
#        batch_acc=net.accuracy(x_batch,t_batch)
#        train_acc=net.accuracy(x_train,t_train)
#        test_acc=net.accuracy(x_test,t_test)
#        print('train_acc,test_acc',train_acc,test_acc)
   # 求损失率
        if i%epoch_num==0:
        
            batch_acc=net.accuracy(x_batch,t_batch)
            train_acc=net.accuracy(x_train,t_train)
            test_acc=net.accuracy(x_test,t_test)
            print('train_acc,test_acc',train_acc,test_acc)
    
        
        
        
    
    
    
    
    
        
        
        
        
        
        
        
        
        
        