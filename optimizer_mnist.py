# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:36:09 2019

@author: HAX
"""

import numpy as np
import matplotlib.pyplot as plt
from optim import SGD,adagrad,momentum,Adam
from collections import OrderedDict
from dataset.mnist import load_mnist
from backwardnet import backnet

def smooth_curve(x):
    """用于使损失函数的图形变圆滑

    参考：http://glowingpython.blogspot.jp/2012/02/convolution-with-numpy.html
    """
    window_len = 11
    s = np.r_[x[window_len-1:0:-1], x, x[-1:-window_len:-1]]
    w = np.kaiser(window_len, 2)
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y[5:len(y)-5]

##定义几种优化optimizer
optimizers=OrderedDict()
optimizers['SGD']=SGD()
optimizers['momentum']=momentum()
optimizers['adagrad']=adagrad()
optimizers['Adam']=Adam()

##提取数据
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)


#设置参数,初始化网络
train_size=x_train.shape[0]
batch_size=100
iter_num=1000

train_loss={}
networks={}
for key in optimizers.keys():
    
    networks[key]=backnet(input_size=784,hidden_size=50,output_size=10)
    ##batch的出现
    train_loss[key]=[]
    
    
##开始训练
for i in range(iter_num):
    #batch制作
    batch_mask=np.random.choice(train_size,batch_size)
    x_batch=x_train[batch_mask]
    t_batch=t_train[batch_mask] 
    
    
    ##梯度下降
    for key in optimizers.keys():
        grads=networks[key].gradient(x_batch,t_batch)
        optimizers[key].update(networks[key].params,grads)
        
        ##损失计算
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
    if i % 100 == 0:
        print( "===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            batch_loss=networks[key].loss(x_batch,t_batch)
            print(key+'batch_loss'+str(batch_loss))
   
#绘制图形
 
markers = {"SGD": "o", "momentum": "x", "adagrad": "s", "Adam": "D"}  
x = np.arange(iter_num)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key) 
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()
    



