# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 16:56:37 2019

@author: HAX
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from multinet_extend import multi_net_extend
from optim import SGD
import matplotlib.pyplot as plt



##数据提取，超参数
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

x_train=x_train[:1000]
t_train=t_train[:1000]

#参数超参
max_epochs = 20
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.01
def __train(weight_init_std):
    #初始化网络结构
    bn_network=multi_net_extend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10, 
                                    weight_init_std=weight_init_std, use_batchnorm=True)
    network=multi_net_extend(input_size=784, hidden_size_list=[100, 100, 100, 100, 100], output_size=10,
                                weight_init_std=weight_init_std)
    
    train_acc_list=[]
    bn_train_acc_list=[]
    
    optimizer=SGD(lr=0.01)
    per_epoch=50
    
    epoch_cnt = 0
    
    for i in range(1000000000):
        ##minibatch
        batch_mask=np.random.choice(train_size,batch_size)
        x_batch=x_train[batch_mask]
        t_batch=t_train[batch_mask]
        
        
        ##梯度下降
        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)
            
        if i %10 == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)
    
            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - " + str(bn_train_acc))
    
            epoch_cnt += 1
            if epoch_cnt >= max_epochs:
                break
    return train_acc_list, bn_train_acc_list
    
##绘图
weight_scale_list = np.logspace(0, -4, num=16)
#print(weight_scale_list)
x = np.arange(max_epochs)
for i, w in enumerate(weight_scale_list):
    print( "============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = __train(w)
    plt.subplot(4,4,i+1)
    plt.title("w:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list, label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle = "--", label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)
        plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')
    
plt.show()
    
    

