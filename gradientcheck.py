# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 10:23:02 2019

@author: HAX
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np
from backwardnet import backnet

from dataset.mnist import load_mnist
(x_train,t_train),(x_test,t_test)=load_mnist(normalize=True,one_hot_label=True)


network=backnet(input_size=784,hidden_size=50,output_size=10)
print(x_train.shape)
x_batch=x_train[:3]
t_batch=t_train[:3]#选取前三行

print('x_batch.shape',x_batch.shape)
grad_numerical=network.numerical_gradient(x_batch,t_batch)
grad_backprop=network.gradiet(x_batch,t_batch)
for key in grad_numerical.keys():
    diff=np.average(np.abs(grad_backprop[key]-grad_numerical[key]))
    print(key+':'+str(diff))


