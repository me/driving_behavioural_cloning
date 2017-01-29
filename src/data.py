import numpy as np
import matplotlib.image as mpimg
import cv2

def process_img(img):
  # orig: 160, 320
  dim = (100,100)
  # cut the top and the bottom
  img = img[35:img.shape[0] - 20, :]
  img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  img = img/255.
  return img

def add_shadow(img):
  top_x1 = 320*np.random.uniform()
  top_x2 = 320*np.random.uniform()
  top_x1, top_x2 = sorted([int(top_x1), int(top_x2)])
  bottom_x1 = 320*np.random.uniform()
  bottom_x2 = 320*np.random.uniform()
  bottom_x1, bottom_x2 = sorted([int(bottom_x1), int(bottom_x2)])
  overlay = img.copy()
  output = img.copy()
  poly = [(bottom_x1, 160), (top_x1, 0), (top_x2, 0), (bottom_x2, 160)]
  cv2.fillConvexPoly(overlay, np.array(poly), (0, 0, 0))
  alpha = 0.8*np.random.uniform()
  cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
  return output

def transform_img(img, steer):
  size = (160, 320)
  base_brightness = 0.1
  tr_range_x = 40
  tr_range_y = 30


  # Brightness
  img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
  brightness = base_brightness + np.random.uniform()
  img[:,:,2] = img[:,:,2]*brightness
  img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

  # Translation

  tr_x = tr_range_x * np.random.uniform() - tr_range_x / 2
  tr_y = tr_range_y * np.random.uniform() - tr_range_y / 2
  steer = steer + tr_x/tr_range_x * .25
  m =  np.float32([[1,0,tr_x],[0,1,tr_y]])
  img = cv2.warpAffine(img, m, size[::-1])

  img = add_shadow(img)

  # Flipping
  flip = np.random.randint(2)
  if flip==0:
      img = cv2.flip(img,1)
      steer = -steer

  return img, steer

def gen_data(driving_log, batch_size=64, augment=True):
  keep_prob_threshold = 0
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

        if abs(steer)>straight_threshold:
          keep_pr = np.random.uniform()
      else:
        path = center
        steer = float(steering)

      if keep_pr > keep_prob_threshold:
        image = mpimg.imread('data/'+path.strip())
        image, steer = transform_img(image, steer)
        image = process_img(image)
        batch_images.append(image)
        batch_angles.append(steer)
    yield np.array(batch_images), np.array(batch_angles)
