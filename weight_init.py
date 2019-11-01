# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 16:14:54 2019

@author: HAX
"""

import numpy as np
import matplotlib.pyplot as plt



def sigmoid(x):
    return 1/(1+np.exp(-x))



##初始数据，网络结构
input_data=np.random.randn(1000,100)
hidden_size=5
node_num=100
activation={}

for i in range(hidden_size):
    if i!=0:
        x=activation[i-1]
        
    w=np.random.randn(node_num,node_num)
    
    a=np.dot(x,w)
    z=sigmoid(a)
    
    activation[i]=z
    
    
##绘制activation的函数
for i in activation.keys():
    plt.subplot(1,len(activation),i+1)
    plt.title(str(i+1)+'__layers')
    if i != 0: plt.yticks([], [])
    plt.hist(activation[i].flatten(),30,range=(0,1))
    
plt.show()