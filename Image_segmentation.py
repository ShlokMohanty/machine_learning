from skimage.color import rgb2gray
import numpy as np 
import cv2
import matplplotlib.pyplot as plt
%matplotlib inline
from scipy import ndimage
image = plt.imread('Thor_Ragnarok.jpeg')
image.shape
plt.imshow(image)
gray = rgb2gray(image)
plt.imshow(gray, cmap='gray')
gray.shape
gray_r = gray.reshape(gray.reshape[0]*gray.shape[1])
for i in range(gray_r.shape[0]):
  if gray_r[i] > gray_r.mean():
    gray_r[i]=1
   else:
    gray_r[i]=0
gray = gray_r.shape(gray.shape[0],gray.shape[1])
plt.imshow(gray, cmap='gray')
