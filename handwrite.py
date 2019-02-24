# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 19:16:02 2019

@author: asus
"""

#------ handwrite digital number recoganize

import sys,os
sys.path.append(os.pardir)  #os.pardir为导入父目录中的文件而服务
from dataset.mnist import load_mnist
(x_train,y_train),(x_test,y_test)=load_mnist(flatten=True,normalize=False)
print(x_train.shape)

