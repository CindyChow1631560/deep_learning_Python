# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 20:08:34 2019

@author: asus
"""
import numpy as np
from common.functions import sigmoid, softmax,cross_entropy_error,numerical_gradient

class TwoLayerNet:
    def __init__(self,input_size,hidden_size,output_size,weight_init=0.1):
        self.params={}
        self.params['w1'] = weight_init*\
            np.random.randn(input_size,hidden_size)
        self.params['b1'] = np.zeros_like(hidden_size)
        self.params['w2'] = weight_init*\
            np.random.randn(hidden_size,output_size)
        self.params['b2'] = np.zeros_like(output_size)
        
    def predict(self,x):
        w1, b1 = self.params['w1'], self.params['b1']
        w2, b2 = self.params['w2'], self.params['b2']
        
        a1 = np.dot(x,w1)+b1
        z1= sigmoid(a1)
        a2= np.dot(z1,w2)+b2
        y = softmax(a2)
        
        return y
    
    def loss(self,x,t):
        y = self.predict(x)
        return cross_entropy_error(y,t)
    
    def accuracy(self,x,t):
        y = self.predict(x)
        y = np.argmax(y,axis=1)
        t = np.argmax(t,axis=1)
        
        au = np.sum(y==t)/float(x.shape(0)) 
        
        return au
    
    def numerical_gradient(self,x,t):
        loss_w = lambda w: self.loss(x,t)
        
        grad={}
        grad['w1'] = numerical_gradient(loss_w, self.prama['w1'])
        grad['b1'] = numerical_gradient(loss_w, self.prama['b1'])
        grad['w2'] = numerical_gradient(loss_w, self.prama['w2'])
        grad['b2'] = numerical_gradient(loss_w, self.prama['b2'])
        
        return grad