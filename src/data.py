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
  camera_offset = 0.25
  while 1:
    for i in range(len(driving_log)):
      center,left,right,steering,throttle,brake,speed = driving_log[i]
      center_img =  process_img(mpimg.imread('data/'+center.strip()))
      left_img =  process_img(mpimg.imread('data/'+left.strip()))
      right_img =  process_img(mpimg.imread('data/'+right.strip()))
      center_steering = float(steering)
      images = np.array([
        center_img, left_img, right_img
      ])
      angles = np.array([
        center_steering, center_steering + camera_offset, center_steering - camera_offset
      ])
      yield images, angles
