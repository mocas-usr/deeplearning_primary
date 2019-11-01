# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 13:55:17 2019

@author: HAX
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from backwardnet import backnet


##提取数据
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)

print('x_train.shape',x_train.shape)


##初始化网络和超参数
network=backnet(input_size=784,hidden_size=50,output_size=10)

train_loss_list=[]
train_acc=[]
train_acc_list=[]
test_acc_list=[]
train_size=x_train.shape[0]#训练数据总量

iter_num=1000##运行次数
epoch=50##每隔50次输出一次准确率
batch_size=100##minibatch
learning_rate=0.1##学习率

##开始运行程序，计算梯度
for i in range(iter_num):
    ##制作minibatch
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask]
    
    #求梯度下降#
    grad=network.gradient(x_batch,t_batch)
    
    for key in ('w1','w2','b1','b2'):
        network.params[key]-=learning_rate*grad[key]
        
    ##求损失
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)
#    train_acc=network.accuracy(x_batch,t_batch)
#    print(train_acc)
    #损失率计算
    if i%epoch==0:
        train_acc=network.accuracy(x_batch,t_batch)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc,test_acc)
print(train_loss_list)
        
    
    
    