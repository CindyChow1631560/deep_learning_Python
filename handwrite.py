# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:16:02 2019

@author: asus
"""

#------ handwrite digital number recoganize

import sys,os
sys.path.append(os.pardir)  #os.pardir为导入父目录中的文件而服务
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
from common.functions import softmax, sigmoid


def image_show(img):
    pil_image= Image.fromarray(np.uint8(img))
    pil_image.show()
    
def get_data():
    (x_train,y_train),(x_test,y_test)=\
    load_mnist(flatten=True,normalize=False,one_hot_label=False)
    return x_test, y_test

def init_network():
    with open("ch03/sample_weight.pkl",'rb') as f:  # rb: 以二进制格式打开一个文件用于只读
      network = pickle.load(f)  #load方法从文件中读取字符串，将它们反序列化成python的数据格式
      
      return network
  
def predict(network,x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1)+b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    y = softmax(a3)
    
    return y

x,t =get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    yp = predict(network,x[i])
    p = np.argmax(yp)
    if p == t[i]:
        accuracy_cnt+=1
        
print("Accuracy is: " + str(accuracy_cnt/len(x)))


#----batch process-------------#

x,t = get_data()
network = init_network()
accu=0
batch = 50
for i in range(0,len(x),batch):
    x_batch = x[i:i+batch]
    y_batch = predict(network,x_batch)
    p = np.argmax(y_batch,axis=1)
    accu += np.sum(p == t[i:i+batch])
    
print("Accuracy: " + str(accu/len(x)))


#(x_train,y_train),(x_test,y_test)=load_mnist(flatten=True,normalize=False)
#img = x_train[0]
#label = y_train[0]
#print(label)
#img=img.reshape(28,28)
#image_show(img)



