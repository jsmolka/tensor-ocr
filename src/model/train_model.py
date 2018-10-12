import numpy as np
import matplotlib.pyplot as plt
import h5py

from keras.datasets import mnist
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras import backend as K

BATCH_SIZE = 128
NUM_CLASSES = 64
EPOCHS = 3
HEIGHT, WIDTH = 128, 128
CHANNELS = 3

h5_file = h5py.File('nist0.h5', 'r')
nist_x = h5_file['nist_x'][:]
nist_y = h5_file['nist_y'][:]

# change when needed
np.random.shuffle(nist_x)

test_samples = int(nist_x.size * 0.1)

test_x = nist_x[:test_samples-1]
train_x = nist_x[test_samples:]

test_y = nist_y[:test_samples-1]
train_y = nist_y[test_samples:]

if K.image_data_format() == 'channels_first':
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[3], train_x[1], train_x[2])
    test_x = test_x.reshape(train_x.shape[0], train_x.shape[3], train_x[1], train_x[2])
    input_shape = (CHANNELS, HEIGHT, WIDTH)
else:
    input_shape = (HEIGHT, WIDTH, CHANNELS)

train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

train_x /= 255
test_x /= 255

mean = np.std(train_x)
train_x -= mean
test_x -= mean

train_y = to_categorical(train_y, NUM_CLASSES, dtype=int)
test_y = to_categorical(test_y, NUM_CLASSES, dtype=int)

