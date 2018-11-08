import numpy as np
import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, LSTM, Flatten
from keras import backend as K
from imageio import imread
from os.path import join, split, splitext

batch_size = 128
epochs = 5
rows, cols = 128, 64
channels = 3


def load_iam_data(path, file_iter, size):
    """Parses images into numpy arrays."""
    x = np.ndarray(shape=(size, cols, rows, channels))
    y = np.ndarray(shape=(size), dtype=object)

    for i in range(0, size):
        fi = next(file_iter)
        x[i] = cv2.imread(join(path, fi))
        y[i] = str(fi)[7:-4]

    return x, y

src = input("Source dir: ")
files = os.listdir(src)
file_iter = iter(files)

validate_x, validate_y = load_iam_data(src, file_iter, 5670)
train_x, train_y = load_iam_data(src, file_iter, batch_size)

if K.image_data_format() == 'channels_first':
    validate_x = validate_x.reshape(validate_x.shape[0], channels, cols, rows)
    train_x = train_x.reshape(train_x.shape[0], channels, cols, rows)
    input_shape = (channels, cols, rows)
else:
    validate_x = validate_x.reshape(validate_x.shape[0], cols, rows, channels)
    train_x = train_x.reshape(train_x.shape[0], cols, rows, channels)
    input_shape = (cols, rows, channels)

validate_x = validate_x.astype('float32')
validate_y = validate_y.astype('str')
validate_x /= 255

train_x = train_x.astype('float32')
train_y = train_y.astype('str')
train_x /= 255

model = Sequential()
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(Conv2D(64, kernel_size=(2,2), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

