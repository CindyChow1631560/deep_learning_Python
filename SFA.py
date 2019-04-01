# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 18:23:39 2019

@author: asus
"""

import numpy as np

class SFA:                                                #slow feature analysis class
    def __init__(self):
        self._Z = []
        self._B = []
        self._eigenVector = []
        
    def getB(self,data):
        self._B = np.matrix(data.T.dot(data))/(data.shape[0]-1)
        
    def getZ(self,data):
        derivativeData = self.makeDiff(data)
        self._Z = np.matrix(derivativeData.T.dot(derivativeData))/(derivativeData.shape[0]-1)    
    
    def makeDiff(self,data):
        diffData = np.mat(np.zeros((data.shape[0],data.shape[1])))
        for i in range(data.shape[1]-1):
            diffData[:,i] = data[:,i] - data[:,i+1]
        diffData[:,-1] = data[:,-1] - data[:,0]
        return np.mat(diffData)         
        
    def fit_transform(self,data,threshold = 1e-7,conponents = -1):
        if conponents == -1:
            conponents = data.shape[0]
        self.getB(data)
        U,s,V = np.linalg.svd(self._B)

        count = len(s)
        for i in range(len(s)):
            if s[i]**(0.5) < threshold:
                count = i
                break
        s = s[0:count]
        s = s**0.5    
        S = (np.mat(np.diag(s))).I
        U = U[:,0:count]
        whiten = S*U.T            
        Z = (whiten*data.T).T
        
        self.getZ(Z)
        PT,O,P = np.linalg.svd(self._Z)

        self._eigenVector = P*whiten
        self._eigenVector = self._eigenVector[-1*conponents:,:]
        
        return data.dot(self._eigenVector.T)