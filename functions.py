# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 17:52:23 2019

@author: asus
"""

import numpy as np
import matplotlib.pylab as plt


def AND(x1,x2):
    w1,w2,theta= 0.5,0.5,0.7
    if x1*w1+x2*w2 <= theta:
        return 0
    elif x1*w1+x2*w2 > theta:
        return 1
        

def sigmoid(x):
    return 1/(1+np.exp(-x))

def step_function(x):
    return np.array(x>0,dtype = np.int)

def ReLU(x):
    return np.maximum(0,x)
    
def identity_function(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x-c)
    exp_sum = np.sum(exp_x)
    y = exp_x/exp_sum
    
    return y   


def mean_squared_error(x,y):
    return 0.5*np.sum((x-y)**2)
    
def cross_entropy_error(x,y):
    delta = 1e-10
    return -np.sum(y*np.log(x+delta))

def numerical_gradient(f,x):
    h = 1e-4
    gradient = np.zeros_like(x)
    
    for i in range(x.size):
        temp = x[i]
        x[i] = temp-h
        fx1 = f(x)
        
        x[i] = temp+h
        fx2 = f(x)
        gradient[i] = (fx2-fx1)/(2*h)
        x[i] = temp
        
    return gradient
    

def gradient_descent(f,init_x,lr=0.01,step=100):
    x = init_x
    
    for i in range(step):
        grad = numerical_gradient(f,x)
        x = -lr*grad
        
    return x
    
#A=np.array([[1,2],[3,4]])
#B=np.array([[5,6],[7,8]])
#C=np.dot(A,B)
#print(C)
#x=np.arange(-5.0,5.0,0.1)
#y=ReLU(x)
#print(y)
#plt.plot(x, y)
#plt.ylim(-1.0,1.0)
#plt.show()
    
""" mini-batch"""

batch_size = 10
train_size = x_train.shape[0]
batch_mask = np.random.choice(train_size,batch_size)
x_batch = x_train[batch_mask]
y_batch = y_train[batch_mask]


""" neural network creation """
def init_network():
    network={}
    network['W1'] = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    return network

def forward(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = identity_function(a3)
    
    return y

network = init_network()
x = np.array([1.0,0.5])
Y = forward(network,x)
y = softmax(Y)
print(y)