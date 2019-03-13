# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 15:04:43 2019

@author: 
    
To represent mini-batch method
"""

import sys,os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from common.functions import softmax, cross_entropy_error

(x_train,y_train),(x_test,y_test) = \
     load_mnist(normalize=True, one_hot_label=True)
     
batch_size = 10
train_size = x_train.shape[0]
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]



class simpleNet:
    def __init__(self):
        self.w = np.random.randn(2,3)
        
    def predict(self,x):
        return np.dot(x,self.w)
    
    def loss(self,x,t):
        z=self.predict(self.w,x)
        y=softmax(z)
        loss=cross_entropy_error(y,t)
        
        return loss
    
net=simpleNet()
print(net.w)