from skimage.color import rgb2gray
import numpy as np 
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
from IPython.display import clear_output 
import matplotlib.pyplot as plt 
import cv2
import matplplotlib.pyplot as plt
%matplotlib inline
from scipy import ndimage
dataset, info =tfds.load('oxford_iiit_pet.*.*', with_info=True)
image = plt.imread('Thor_Ragnarok.jpeg')
def normalize(input_image, input_mask):
  input_image = tf.cast(input_image, tf.float32) / 255.0
  input_mask -= 1
  return input_image, input_mask
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
gray = rgb2gray(image)
gray_r=gray.reshape(gray.shape[0]*gray.shape[1])



