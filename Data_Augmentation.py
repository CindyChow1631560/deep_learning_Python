# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:01:39 2019

@author: asus
"""

'''
 Data Augmentation还可以通过其他各种方法扩充图像，
 比如裁剪图像的“crop处理”、将图像左右翻转的"filp处理” 等。
 对于一般的图像，施加亮度等外观上的变化、放大缩小等尺度上的变化也是有效的
 此外，还可以修改图片的颜色
'''

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

image = Image.open("C:/Users/asus/Pictures/love.jpg")
img = np.array(image)
plt.figure()
plt.imshow(img)
plt.show()
# image crop
box=(100,100,500,500)
imgag = image.crop(box)
plt.figure()
plt.imshow(imgag)

#print(image.size)
HEIGHT= image.size[0]  ## actually, WIDTH should be image.size[0]
WIDTH = image.size[1]

# change color
for i in range(HEIGHT):
    for j in range(WIDTH):
        data = image.getpixel((i,j)) #获取该图片的所有像素点
        if(data[0]>100 and data[1]>100 and data[2]>100): # 如果RGBA值中的RGB值均大于100
            img1 = image.putpixel((i,j),(255,210,10,0)) # 将这些像素点改为红色
plt.figure()
plt.show(img1)


# Horizontal flip
flip_img = np.fliplr(img)
plt.figure()
plt.imshow(flip_img)
plt.show()


# shifting left
for i in range(HEIGHT, 1, -1):
  for j in range(WIDTH):
     if (i < HEIGHT-20):
       img[j][i] = img[j][i-20]
     elif (i < HEIGHT-1):
       img[j][i] = 0
plt.figure()
plt.imshow(img)
plt.show()

# shifting right
for i in range(HEIGHT):
  for j in range(WIDTH):
     if (i < HEIGHT-20):
       img[j][i] = img[j][i+20]
#     elif (i < HEIGHT-1):
#       img[j][i] = 0
plt.figure()
plt.imshow(img)
plt.show()

# shifting up
for i in range(WIDTH):
    for j in range(HEIGHT):
        if (i+20<WIDTH):
            img[i][j] = img[i+20][j]
plt.figure()
plt.imshow(img)
plt.show()

# shifting down
for i in range(WIDTH,1,-1):
    for j in range(HEIGHT):
       if(i>20 and i<WIDTH):
            img[i][j] = img[i-20][j]
       elif(i<20):
           img[i][j] = 0
plt.figure()
plt.imshow(img)
plt.show()

# ADD Noise
noise = np.random.randint(2,dtype = 'uint8')
for i in range(WIDTH):
    for j in range(HEIGHT):
        img[i][j] = img[i][j]+noise
        
plt.figure()
plt.imshow(img)
plt.show()

