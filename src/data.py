import numpy as np
import matplotlib.image as mpimg
import cv2

def process_img(img):
  # orig: 160, 320
  dim = (100,100)
  # cut the top part -> 100, 320
  img = img[35:img.shape[0] - 20, :]
  img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  #img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
  img = img/255.
  return img

def transform_img(img, steer):
  size = (160, 320)

  # Flipping
  flip = np.random.randint(2)
  if flip==0:
      img = cv2.flip(img,1)
      steer = -steer

  # # Translation
  # translation_range_x = 15
  # translation_range_y = 30
  # tr_x = translation_range_x*np.random.uniform()-translation_range_x/2
  # steer = steer + tr_x/translation_range_x*2*.2
  # tr_y = translation_range_y*np.random.uniform()-translation_range_y/2
  # translation_matrix = np.float32([[1,0,tr_x],[0,1,tr_y]])
  # img = cv2.warpAffine(img,translation_matrix,size)

  return img, steer

def gen_data(driving_log, batch_size=64, augment=True):
  reject_prob_threshold = 0
  straight_threshold = .1
  camera_offset = 0.25
  size = len(driving_log)
  while 1:
    batch_images = []
    batch_angles = []
    while len(batch_images) < batch_size:
      index = np.random.randint(size)
      center,left,right,steering,throttle,brake,speed = driving_log[index]
      path = None
      steer = None
      keep_pr = 1

      if augment:
        camera = np.random.randint(3)
        if camera == 0:
          path = left
          steer = float(steering) + camera_offset
        elif camera == 1:
          path = right
          steer = float(steering) - camera_offset
        else:
          path = center
          steer = float(steering)

        if abs(steer)<straight_threshold:
          keep_pr = np.random.uniform()
      else:
        path = center
        steer = float(steering)

      if keep_pr > reject_prob_threshold:
        image = mpimg.imread('data/'+path.strip())
        image, steer = transform_img(image, steer)
        image = process_img(image)
        batch_images.append(image)
        batch_angles.append(steer)
    yield np.array(batch_images), np.array(batch_angles)
