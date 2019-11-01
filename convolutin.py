# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 16:10:01 2019

@author: HAX
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 21:52:03 2019

@author: HAX
"""
import numpy as np

def im2col(input_data,filter_h,filter_w,stride=1,pad=0):
    """

    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col 2维数组
    """
    N,C,H,W=input_data.shape
    out_h=(H+2*pad-filter_h)//stride+1
    out_w=(W+2*pad-filter_w)//stride+1
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col=np.zeros((N,C,filter_h,filter_w,out_h,out_w))
    
    for y in range(filter_h):
        y_max= y + stride*out_h
        for x in range(filter_w):
            x_max=x+stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            
            
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col
    
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]
    



class convolution:
    def __init__(self,w,b,stride=1,pad=0):
        self.w=w
        self.b=b
        self.stride=stride
        self.pad=pad
         # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_w = None
        
        # 权重和偏置参数的梯度
        self.dw = None
        self.db = None
    def forward(self,x):
#        FN,C,FH,FW=self.w.shape
#        N,C,H,W=x.shape
#        out_h=int(1+(H+2*self.pad-FH)/self.stride)
#        out_w=int(1+(W+2*self.pad-FW)/self.stride)
#        
#        col=im2col(x,FH,FW,self.stride,self.pad)
#        col_w=self.w.reshape(FN,-1).T
#        out=np.dot(col,col_w)+self.b
#        
#        out=out.reshape(N,out_h,out_w,-1).transpose(0,3,1,2)
#        
#        return out
        FN, C, FH, FW = self.w.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_w = self.w.reshape(FN, -1).T

        out = np.dot(col, col_w) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_w = col_w

        return out
    def backward(self, dout):
        FN, C, FH, FW = self.w.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dw = np.dot(self.col.T, dout)
        self.dw = self.dw.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_w.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx



class pooling:
    def __init__(self,pool_h,pool_w,stride=1,pad=0):
        self.pool_h=pool_h
        self.pool_w=pool_w
        self.stride=stride
        self.pad=pad
        
    def forward(self,x):
        N,C,H,W=x.shape
        out_h=int(1+(H-self.pool_h)/self.stride)
        out_w=int(1+(W-self.pool_w)/self.stride)
        
        ##展开
        col=im2col(x,self.pool_h,self.pool_w,self.stride,self.pad)
        col=col.reshape(-1,self.pool_h*self.pool_w)
        
        #最大值
        arg_max = np.argmax(col, axis=1)
        out=np.max(col,axis=1)
        out=out.reshape(N,out_h,out_w,C).transpose(0,3,1,2)
        self.x = x
        self.arg_max = arg_max
        return out
    
    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx
        
        
        