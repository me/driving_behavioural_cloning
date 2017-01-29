# Self-driving network by behavioural cloning

This is an implementation of a self-driving convolutional neural network.

The network is based on the NVIDIA
[End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf) paper.
It has the following architecture:

* 24 5x5 convolutions, with  2x2 strides, valid padding, ELU activation.
* 36 5x5 convolutions, with  2x2 strides, valid padding, ELU activation.
* 48 5x5 convolutions, with  2x2 strides, valid padding, ELU activation.
* 64 3x3 convolutions, with  1x1 strides, valid padding, ELU activation.
* 64 3x3 convolutions, with  1x1 strides, valid padding, ELU activation.
* 100 node fully connected layer, with a 0.5 dropout, ELU activation.
* 50 node fully connected layer, with a 0.5 dropout, ELU activation.
* 10 node fully connected layer, with a 0.5 dropout, ELU activation.

The network is trained on the driving log and images provided by Udacity, using 15 epochs,
a batch size of 24108, and the Adam optimizer with a learning rate of 0.001.

Only 10% of the data set of the initial set is used for validation: this is in order to make full
use of the available data, and considering that since the real test is how well the network generalises
to driving on the provided tracks, the validation is mostly used as a control.

The images are first processed by removing the top and bottom part which don't generally
contain useful information; they are then resized to 100x100 pixels, and locally normalized.

The model is then run on an image generator that augments the images in the dataset.
The generator selects an image at random from the original set: it then chooses either the
center, left, or right camera image at random, adding or subtracting 0.25 to the steering angle
if the left or right cameras are used.

The image then goes through a series of random transformations:

* The brightness of the image is randomly reduced, to help the network generalise to lower-light situations.
* The images are randomly translated on the x and y axes, in order to augment the training data when steering is
  required. For each pixel of x translation, the steering angle is increased or reduced by 0.25.
* A random polygon is overlayed on top of the image, in order to simulate shadows.
* The image is randomly flipped horizontally, in order to augment the number of examples steering in the
  opposite direction.


This setup allows the network to drive comfortably on both tracks!

On the first track, the network is able to drive indefinitely with a base throttle of 0.2, reduced based
on the steering angle. It exibits a swerving behaviour that at higher speeds gets out of control, however.

Interestingly, the network seems to drive more smoothly on the second track, where it's able to complete the
course with a base throttle of 0.3. Maybe the data augmentation is too aggressive and leads the network
to underfit on the first track; however, all the other techniques I tried seemed to lead to worse results.
More experimentation would be needed on this.


To run:

```
$ python drive.py outputs/steering_model/steering_angle.json
```

Note: for the second track, open drive.py and change BASE_THROTTLE to 0.3, or the car won't be able
to get over the hills in the track.

To retrain: place driving_log.csv and images in `data`, and run

```
$ python src/model.py
```

