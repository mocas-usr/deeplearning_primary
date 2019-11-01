# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 19:35:46 2019

@author: HAX
"""
##mnist数据集的使用，下载数据集使用##
from mnist import load_train_images,load_train_labels,load_test_images,load_test_labels
from PIL import Image 
import matplotlib.pyplot as plt 
import numpy as np
##sigmoid函数##
def sigmoid(a):
    return 1/(1+np.exp(-a))
##数据集，测试集获取函数##
def init_mnist():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    return train_images, train_labels,test_images,test_labels
##初始化权重参数函数##
##初始化参数的维度要对应##
def init_network():
    network={}
    network['w1']=np.random.randn(784,50)
    network['b1']=np.ones(50)
    network['w2']=np.random.randn(50,100)
    network['b2']=np.ones(100)
    network['w3']=np.random.randn(100,10)
    network['b3']=np.ones(10)
    return network
##softmax层函数##
def softmax(x):
    c=np.max(x)
    exp_xc=np.exp(x-c)
    sum_exp_xc=np.sum(exp_xc)
    y=exp_xc/sum_exp_xc
    return y
##预测函数##
def predict(network,x):
    w1,w2,w3=network['w1'],network['w2'],network['w3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    a1=np.dot(x,w1)+b1
    z1=sigmoid(a1)
    a2=np.dot(z1,w2)+b2
    z2=sigmoid(a2)
    a3=np.dot(z2,w3)+b3
    z3=sigmoid(a3)
    y=softmax(z3)
    return y

if __name__=='__main__':
    
    ##主程序##
    train_images, train_labels,test_images,test_labels=init_mnist()#提取数据集
    print(train_images.shape)
    train_x=train_images.reshape((60000,784))
    label_t=train_labels
    print('train_x成为二维数组的shape',train_x.shape)
    img_0=train_x[0]
    img0=train_images[0]                                            #提取第一张图片
    label0=train_labels[0]                                      #提取第一个标签     
    print(img0.shape)
    plt.imshow(img0,cmap='gray')                                #显示图片
    #pil_img.show(img0)
    print(label0)
    img0=img0.reshape(784)
    print('重新reshape之后的img0',img0.shape)
    network=init_network()
    y=predict(network,img0)
    print(y)
    num=np.argmax(y)
    print(num)
    
    ##使用batch_size##
    batch_size=100          ##批数量
    accuracy_int=0
    for i in range(0,len(train_x),batch_size):
        x_batch=train_x[i:i+batch_size]
        y_batch=predict(network,x_batch)
        p=np.argmax(y_batch,axis=1)
        accuracy_int+=np.sum(p==label_t[i:i+batch_size])   
    print('y_batch.shape',y_batch.shape)
    print('accuracy'+str(float(accuracy_int)/len(train_x)))


    
    
    