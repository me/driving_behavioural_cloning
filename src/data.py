import numpy as np
import matplotlib.image as mpimg
import cv2

def process_img(img):
  dim = (160,80)
  img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  img = img/255.
  return img

def gen(driving_log):
  print(driving_log.shape)
  while 1:
    for i in range(len(driving_log)):
      center,left,right,steering,throttle,brake,speed = driving_log[i]
      center_img = mpimg.imread('data/'+center)
      img = process_img(center_img)
      yield np.array([img]), np.array([steering])
