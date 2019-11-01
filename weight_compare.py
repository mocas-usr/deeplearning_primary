# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:19:45 2019

@author: HAX
"""

import sys,os 
sys.path.append(os.pardir)
import numpy as np
from mul_layer_net import multi_layer_net
from dataset.mnist import load_mnist
from optim import SGD
import matplotlib.pyplot as plt



##初始化数据，超参

(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

#weight_type={'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
weight_type = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}
train_size=x_train.shape[0]
batch_size=100
iter_num=1000
epoch_num=50
train_loss={}
networks={}
optimizer=SGD(lr=0.1)

##神经网络结构
hidden_size=5
node_num=100

#初始化网络结构
for key,weight in weight_type.items():
    networks[key]=multi_layer_net(input_size=784,hidden_layer_list=[100,100,100,100],output_size=10,weight_init_std=weight)
    train_loss[key]=[]
    
    
for i in range(iter_num):
    ##取batch数据
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]
    #计算梯度
    for key in weight_type.keys():
        grads=networks[key].gradient(x_batch,t_batch)
        optimizer.update(networks[key].params,grads)
    
    #计算损失
        batch_loss=networks[key].loss(x_batch,t_batch)
        train_loss[key].append(batch_loss)
        
        
        if i%100==0:
            print('+++++++++++')
            for key in weight_type.keys():
                loss = networks[key].loss(x_batch, t_batch)
                print(key + ":" + str(loss))


#绘图
makers={'std=0.01':'o','Xavier':'s','He':'D'}
x=np.arange(iter_num)
for key in weight_type.keys():
    plt.plot(x,train_loss[key],marker=makers[key], markevery=100, label=key)

plt.xlabel('liter_num')
plt.ylabel('loss')
plt.legend()
plt.show()