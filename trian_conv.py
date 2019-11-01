# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 21:36:55 2019

@author: HAX
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from simpleconnet import simpleconnet
from train_step import Trainer

#读入数据
(x_train,t_train),(x_test,t_test)=load_mnist(flatten=False)
x_train, t_train = x_train[:5000], t_train[:5000]

##参数设置
max_epoch=20

network=simpleconnet(input_dim=(1,28,28),conv_param={'filter_num':30,'filter_size':5,
                    'pad':0,'stride':1},hidden_size=100,output_size=10,weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epoch, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()
network.save_params(file_name='params.pkl')

