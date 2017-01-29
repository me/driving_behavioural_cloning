#!/usr/bin/env python
"""
Steering angle prediction model
"""
import os
import argparse
import json
import csv
import numpy as np
from data import gen_data

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D
from keras.optimizers import Adam


def get_model(time_len=1):
  ch, row, col = 3, 100, 100  # input format

  model = Sequential()


  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", input_shape=(row, col, ch)))
  model.add(ELU())
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
  model.add(ELU())
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
  model.add(ELU())

  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
  model.add(ELU())

  model.add(Flatten())

  model.add(Dense(100))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(50))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(10))
  model.add(Dropout(.5))
  model.add(ELU())


  model.add(Dense(1, activation='tanh', name='output'))

  optimizer = Adam(lr=0.001)
  model.compile(optimizer=optimizer, loss="mse")

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--epoch', type=int, default=15, help='Number of epochs.')
  args = parser.parse_args()

  with open('data/driving_log.csv', 'r') as f:
    driving_log = np.array(list(csv.reader(f))[1:])

  np.random.shuffle(driving_log)

  train_proportion = 0.9
  train_n = int(train_proportion*len(driving_log))

  driving_log_train = driving_log[:train_n]
  driving_log_valid = driving_log[train_n:]

  epochsize = len(driving_log)

  model = get_model()
  model.fit_generator(
    gen_data(driving_log_train),
    samples_per_epoch=epochsize*3,
    nb_epoch=args.epoch,
    validation_data=gen_data(driving_log_valid, augment=False),
    nb_val_samples=len(driving_log_valid)
  )
  print("Saving model weights and configuration file.")

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.h5", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

  import gc; gc.collect()
