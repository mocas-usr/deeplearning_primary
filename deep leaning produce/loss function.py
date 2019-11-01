# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 09:10:17 2019

@author: HAX
"""
##均方误差损失函数##
import numpy as np
def  mean_squared_error(y,t):
    e=0.5*np.sum((y-t)**2)
    return e
##交叉熵误差##
def cross_entropy_error0(y,t):
    delta=1e-7#1*10的-7次幂
    e=-np.sum(t-np.log(y+delta))
    return e
##mini_batch交叉熵实现##
def cross_entropy_error1(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape[0]
    e=-np.sum(t*np.log(y+1e-7))/batch_size
    return e
###非hot形式的交叉熵函数##
def cross_entropy_error(y,t):
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
    batch_size=y.shape(0)
    e=-np.sum(np.log(y[np.arrange(batch_size),t]+1e-7))/batch_size
    return e


##mini_batch的应用学习#
import sys,os
#sys.path.append(os.pardir)
#print(os.getcwd())
#os.path.abspath(os.path.dirname(os.getcwd()))
#print(os.getcwd())
path=os.path.dirname(os.getcwd())
os.chdir(path) 
#import mod1
#import mod2.mod2
import numpy as np
from mnist import load_train_images,load_train_labels,load_test_images,load_test_labels
##数据集，测试集获取函数##
def init_mnist():
    train_images = load_train_images()
    train_labels = load_train_labels()
    test_images = load_test_images()
    test_labels = load_test_labels()
    return train_images, train_labels,test_images,test_labels

train_images, train_labels,test_images,test_labels=init_mnist()#提取数据集
train_x=train_images.reshape((60000,784))
train_t=train_labels
print('train_x.shape',train_x.shape)
print('train_t.shape',train_t.shape)
train_size=train_x.shape[0]
batch_size=10
batch_mask=np.random.choice(train_size,batch_size)
x_batch=train_x[batch_mask]
t_batch=train_t[batch_mask]

##微分函数##
def numerical_diff(f,x):
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

##定义一个普通函数##
def function_2(x):
    return x[0]**2+x[1]**2
   

##求梯度函数##
def numberical_gradient(f,x):
    h=1e-4
    grad=np.zeros_like(x)
    for idx in range(x.size):
        tmp_val=x[idx]
        x[idx]=tmp_val+h
        fxh1=f(x)           #f(x+h)
        
        x[idx]=tmp_val-h
        fxh2=f(x)
        grad[idx]=(fxh1-fxh2)/(2*h)
        x[idx]=tmp_val
    return grad
function_2(np.array([1,2]))
#numberical_diff(function_2,np.array([1,2])
numberical_gradient(function_2,np.array([3.0,4.0]))

##梯度下降函数##
def gradient_descent(f,init_x,lr=0.01,step_num=100):
    x=init_x
    for i in range(step_num):
        grad=numberical_gradient(function_2,x)
        x=x-lr*grad
    return x


##求函数的最小值##
def function_3(x):
    y=x[0]**2+x[1]**2
    return y
##利用梯度法求最小值时x##
final_x=gradient_descent(function_3,np.array([1.0,2.0]),lr=0.01,step_num=2000)
print(final_x)
y=function_3(final_x)
        

#import sys,os
#print(os.getcwd())
#print(sys.argv[0])
#print(__file__)
#print(os.path.abspath('.'))
#dirname, filename = os.path.split(os.path.abspath(sys.argv[0])) 
#os.path.realpath(sys.argv[0]) 
#dirname, filename = os.path.split(os.path.abspath(__file__)) 
#os.path.realpath(__file__) 
#sys.path.append(os.pardir) #添加路径到这目录里面
#print(os.getcwd())
#import numpy as np
#os.chdir(os.getcwd())
#print(os.curdir)
#path=os.path.dirname(os.getcwd())
#sys.path.append(path)
#
#
#import os
#import sys
# 
#print(sys.path)
#print(sys.path[0])
#

#import os
#import sys
#print("abs path is %s" %(os.path.abspath(sys.argv[0])))
#import os
# 
## Build paths inside the project like this: os.path.join(BASE_DIR, ...)
#BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('__file__')))
#print("__file Output:",'__file__')
#print(os.path.abspath('__file__'))
#print(os.path.dirname('__file__'))
#os.path.split(os.path.realpath('__file__'))[0]
#
#print()

##实现simple类的网络##
import sys,os
sys.path.append(os.pardir)
import numpy as np
os.chdir(os.path.dirname(os.getcwd()))
from shuzishibie import softmax

class simplenet:
    def __init__(self):
        self.w=np.random.randn(2,3)##高斯分布进行初始化
    def predict(self,x):
        return np.dot(x,self.w)
    def loss(self,x,t):
        z=self.predict(x)
        y=softmax(z)
        loss=cross_entropy_error0(y,t)
        
        return loss
##求梯度函数##
def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
        it.iternext()   
        
    return grad
##简单实现##
net=simplenet()
print(net.w)
x=np.array([0.6,0.9])
p=net.predict(x)
print(p)
num=np.argmax(p)
t=np.array([0,0,1])
net.loss(x,t)


##有问题未解决##
def f(w):
    return net.loss(x,t)
print(f(net.w))
dw=numberical_gradient(f,net.w)
print(dw)
    
    
