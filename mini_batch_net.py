# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:06:15 2019

@author: HAX
"""

import numpy as np
from dataset.mnist import load_mnist
from twolayernet import two_layer_net
import matplotlib.pyplot as plt

##数据提取##
(train_x, train_t), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
print('t_train.shape',train_t.shape)
print(train_t)

#train_test=load_test_images().reshape((10000,784))
#train_labels=load_test_labels()


#超参数#
iter_num=10000
train_size=train_x.shape[0]
batch_size=100
learning_rate=0.1
##loss损失和学习情况##
train_loss_list=[]
train_acc_list=[]
test_acc_list=[]
iter_per_epoch=max(train_size/batch_size,1)


network=two_layer_net(input_size=784,hidden_size=50,output_size=10)
##获取minibatch,每次抽一百数据，循环iter_num次数##
for i in range(iter_num):
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=train_x[batch_mask]
    t_batch=train_t[batch_mask]
#    print(x_batch.shape)
#    print(t_batch.shape)
    #grad=network.numerical_gradient(x_batch,t_batch)
    grad = network.gradient(x_batch, t_batch)
    
    for key in ('w1','w2','b1','b2'):
        network.params[key]=network.params[key]-learning_rate*grad[key]
     
    ##记录学习过程##
    loss=network.loss(x_batch,t_batch)
    train_loss_list.append(loss)        
    
    ##计算每个epoch的识别精度
    if i%iter_per_epoch==0:
        train_acc=network.accuracy(train_x,train_t)
        test_acc=network.accuracy(x_test,t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('train_acc,test_acc'+str(train_acc)+','+str(test_acc))


#绘制图形##   
maker={''}
x=np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')##绘图
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")#xy轴的坐标
plt.ylabel("accuracy")
plt.ylim(0, 1.0)#设置y坐标轴的范围
plt.legend(loc='lower right')#图例
plt.show()
    