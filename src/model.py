#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
import csv
import matplotlib.image as mpimg
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

def process_img(img):
  dim = (160,80)
  img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = img.reshape(img.shape  + (1,))
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


def get_model(time_len=1):
  ch, row, col = 1, 80, 160  # input format

  model = Sequential()
  model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same", input_shape=(row, col, ch)))
  model.add(ELU())
  model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(ELU())
  model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
  model.add(Flatten())
  model.add(Dropout(.2))
  model.add(ELU())
  model.add(Dense(512))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(1))

  model.compile(optimizer="adam", loss="mse")

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--batch', type=int, default=128, help='Batch size.')
  parser.add_argument('--epoch', type=int, default=5, help='Number of epochs.')
  parser.add_argument('--epochsize', type=int, default=0, help='How many frames per epoch.')
  parser.add_argument('--skipvalidate', dest='skipvalidate', action='store_true', help='Multiple path output.')
  parser.set_defaults(skipvalidate=False)
  parser.set_defaults(loadweights=False)
  args = parser.parse_args()

  with open('data/driving_log.csv', 'r') as f:
    driving_log = np.array(list(csv.reader(f))[1:])

  np.random.shuffle(driving_log)

  train_proportion = 0.8
  train_n = int(train_proportion*len(driving_log))

  driving_log_train = driving_log[:train_n]
  driving_log_valid = driving_log[train_n:]


  epochsize = args.epochsize
  if args.epochsize == 0:
    epochsize = len(driving_log)

  if epochsize > len(driving_log):
    print("Epoch must be less than {}".format(len(driving_log)))
    quit()

  gen(driving_log)

  model = get_model()
  model.fit_generator(
    gen(driving_log_train),
    samples_per_epoch=epochsize,
    nb_epoch=args.epoch,
    validation_data=gen(driving_log_valid),
    nb_val_samples=len(driving_log_valid)
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.keras", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)
