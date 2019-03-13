# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 11:53:20 2019

@author: asus
"""

import numpy as np

class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.momentum = momentum
        self.lr = lr
        self.v = None
        
    def update(self,params,grad):
        if self.v is None:
            self.v = {}
            for key, item in params.items():
                self.v[key] = np.zeros_like(item)
                
        for key in params.items():
            self.v[key] = self.v[key]*self.momentum-self.lr*self.grads[key]
            params[key] += self.v[key]
            
            
class AdaGrad:
    def __init__(self,lr=0.01):
        self.lr=lr
        self.h=None
        
    def update(self,params,grads):
        if self.h is None:
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.items():
            self.h[key] = self.h[key] + self.grads[key]*self.grads[key]
            params[key] =  params[key] - self.lr*1/(np.sqrt(self.h[key]+1e-7))*grads[key]
    