# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 10:43:48 2019

@author: HAX
"""

import sys,os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from simpleconnet import simpleconnet
from matplotlib.image import imread
from convolutin import convolution
def filter_show(filters, nx=4, show_num=16):
    
    
    FN,FC,FH,FW=filters.shape
    ny = int(np.ceil(show_num / nx))
    fig=plt.figure()
    
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(show_num):
        ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
        ax.imshow(filters[i, 0], cmap=plt.cm.gray_r, interpolation='nearest')
        
network=simpleconnet(input_dim=(1,28,28),conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                     hidden_size=100, output_size=10, weight_init_std=0.01)
network.load_params(file_name='params.pkl')
filter_show(network.params['w1'], 16)


img=imread('./dataset/lena_gray.png')
print(img.shape)
img=img.reshape(1,1,*img.shape)
print(img.shape)
fig=plt.figure()
w_idx = 1

for i in range(16):
    w = network.params['w1'][i]
    b = 0  # network.params['b1'][i]

    w = w.reshape(1, *w.shape)
    #b = b.reshape(1, *b.shape)
    conv_layer = convolution(w, b) 
    out = conv_layer.forward(img)
    out = out.reshape(out.shape[2], out.shape[3])
    
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.imshow(out, cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()