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


def get_model():
  """
  Constructs the network
  """

  # input format
  ch, row, col = 3, 100, 100

  model = Sequential()

  # 3 convolution layers + ELU
  model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode="valid", input_shape=(row, col, ch)))
  model.add(ELU())
  model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode="valid"))
  model.add(ELU())
  model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode="valid"))
  model.add(ELU())

  # 2 convolution layers + ELU
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
  model.add(ELU())
  model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode="valid"))
  model.add(ELU())

  # Flatten
  model.add(Flatten())

  # Fully connected layers
  model.add(Dense(100))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(50))
  model.add(Dropout(.5))
  model.add(ELU())
  model.add(Dense(10))
  model.add(Dropout(.5))
  model.add(ELU())

  # Activation layer
  model.add(Dense(1, activation='tanh', name='output'))

  optimizer = Adam(lr=0.001)
  model.compile(optimizer=optimizer, loss="mse")

  return model


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Steering angle model trainer')
  parser.add_argument('--epoch', type=int, default=10, help='Number of epochs.')
  args = parser.parse_args()

  # Extract driving_log into a list of lists
  with open('data/driving_log.csv', 'r') as f:
    driving_log = np.array(list(csv.reader(f))[1:])

  np.random.shuffle(driving_log)

  # Split into training and validation set
  train_proportion = 0.9
  train_n = int(train_proportion*len(driving_log))

  driving_log_train = driving_log[:train_n]
  driving_log_valid = driving_log[train_n:]

  # The number of training examples.
  train_set_size = len(driving_log)

  model = get_model()
  model.fit_generator(
    gen_data(driving_log_train), # image generator from the training set
    samples_per_epoch=train_set_size*3, # set epoch size
    nb_epoch=args.epoch, # number of epochs
    validation_data=gen_data(driving_log_valid, augment=False), # validation data
    nb_val_samples=len(driving_log_valid) # size of the validation set
  )
  print("Saving model weights and configuration file.")

  # Save output

  if not os.path.exists("./outputs/steering_model"):
      os.makedirs("./outputs/steering_model")

  model.save_weights("./outputs/steering_model/steering_angle.h5", True)
  with open('./outputs/steering_model/steering_angle.json', 'w') as outfile:
    json.dump(model.to_json(), outfile)

  # Avoids occasional error on exit
  import gc; gc.collect()
