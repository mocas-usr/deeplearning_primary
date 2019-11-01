# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 15:06:07 2019

@author: HAX
"""

import numpy as np 
from optim import SGD,momentum,adagrad,Adam
from collections import OrderedDict
import matplotlib.pyplot as plt


#3定义f函数，和导数
def f(x,y):
    return x**2/20.0+y**2

def df(x,y):
    return x/10.0,y*2


##初始值给定,
init_pos=(-7.0,2.0)
params={}
params['x']=init_pos[0]
params['y']=init_pos[1]

grads={}
grads['x']=0
grads['y']=0

optimizers=OrderedDict()##有序字典
optimizers['SGD']=SGD(lr=0.95)
optimizers['momentum']=momentum(lr=0.1)
optimizers['adagrad']=adagrad(lr=1.5)
optimizers['Adam']= Adam(lr=0.3)

idx=1##图的位置分布
for key in optimizers.keys():##取每个key键值
        ##尝试每种优化方法
        ##初始化参数
    optimizer=optimizers[key]
    params['x']=init_pos[0]
    params['y']=init_pos[1]
    x_history=[]
    y_history=[]
    
    ##定义梯度的来源以及梯度下降过程
   
    for i in range(30):
        x_history.append(params['x'])
        y_history.append(params['y'])
        
        grads['x'],grads['y']=df(params['x'],params['y'])
        optimizer.update(params,grads)
        
        
    ##绘图
    ##坐标x，y
    
    x=np.arange(-10,10,0.01)
    y=np.arange(-5,5,0.01)
    ##生成网状图
    X, Y = np.meshgrid(x, y) 
    Z = f(X, Y)
     # for simple contour line  
    mask = Z > 7
    Z[mask] = 0
    plt.subplot(2,2,idx)
    idx+=1
    plt.plot(x_history,y_history,'o-',color='red')
    plt.contour(X,Y,Z)
    plt.ylim(-10.10)
    plt.xlim(-10.10)
    plt.plot(0,0,'+')
    plt.title(key)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()

plt.show()    
        
