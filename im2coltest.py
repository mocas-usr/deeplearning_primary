# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 15:41:07 2019

@author: HAX
"""
import sys,os
sys.path.append(os.pardir)
from im2col import im2col
import numpy as np

x1=np.random.rand(1,3,7,7)
col1=im2col(x1,5,5,stride=1,pad=0)
print(col1.shape)

x2=np.random.rand(10,3,7,7)
col2=im2col(x2,5,5,stride=1,pad=0)
print(col2.shape)

