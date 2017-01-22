import numpy as np
import matplotlib.image as mpimg
import cv2

def process_img(img):
  # orig: 160, 320
  dim = (160,50)
  # cut the top part -> 100, 320
  img = img[60:img.shape[0], :]
  img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = img/255.
  return img

def gen_data(driving_log):
  print(driving_log.shape)
  while 1:
    for i in range(len(driving_log)):
      center,left,right,steering,throttle,brake,speed = driving_log[i]
      center_img = mpimg.imread('data/'+center)
      img = process_img(center_img)
      yield np.array([img]), np.array([float(steering)])
