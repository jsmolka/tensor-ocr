import numpy as np
import os
import cv2
import keras
from keras.models import Sequential, Model
from keras.layers import (Conv2D, MaxPooling2D, LSTM, Flatten, 
TimeDistributed, Activation, Dense, Input, Lambda, Bidirectional, 
Dropout, Reshape)
from keras import backend as K
from os.path import join, split, splitext

batch_size = 128
epochs = 5
rows, cols = 128, 64
channels = 1


def load_iam_data(path, file_iter, size):
    """Parses images into numpy arrays."""
    x = np.ndarray(shape=(size, rows, cols))
    y = np.ndarray(shape=(size), dtype=object)

    for i in range(0, size):
        fi = next(file_iter)

        tmp = cv2.imread(join(path, fi), cv2.IMREAD_GRAYSCALE)
        tmp = tmp.reshape(rows, cols)

        y[i] = str(fi)[7:-4]

    return x, y

def ctc_lambda(args):
    pred_y, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, pred_y, input_length, label_length)

src = input("Source dir: ")
files = os.listdir(src)
file_iter = iter(files)

validate_x, validate_y = load_iam_data(src, file_iter, 1)
train_x, train_y = load_iam_data(src, file_iter, 5)

if K.image_data_format() == 'channels_first':
    validate_x = validate_x.reshape(validate_x.shape[0], channels, rows, cols)
    train_x = train_x.reshape(train_x.shape[0], channels, rows, cols)
    input_shape = (channels, rows, cols)
else:
    validate_x = validate_x.reshape(validate_x.shape[0], rows, cols, channels)
    train_x = train_x.reshape(train_x.shape[0], rows, cols, channels)
    input_shape = (rows, cols, channels)

input_shape = (rows, cols)

validate_x = validate_x.astype('float32')
validate_y = validate_y.astype('str')
validate_x /= 255

train_x = train_x.astype('float32')
train_y = train_y.astype('str')
train_x /= 255

input_shape = (None, rows, cols, channels)
input_data = Input(name="input1", shape=input_shape)
model = TimeDistributed(Conv2D(64, kernel_size=(3,3), activation='relu'))(input_data)
model = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(model)
model = TimeDistributed(Conv2D(64, kernel_size=(3,3), activation='relu'))(model)
model = TimeDistributed(Conv2D(64, kernel_size=(3,3), activation='relu'))(model)
model = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(model)
model = TimeDistributed(Conv2D(64, kernel_size=(3,3), activation='relu'))(model)
model = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(model)

conv_to_rnn_dims = (cols // 4, (rows // 4) * 64)
model = Reshape(target_shape=conv_to_rnn_dims)(model)

model = LSTM(512)(model)
pred_y = Dense(78, kernel_initializer='he_normal')(model)
# pred_y = Activation('softmax')(model)

labels = Input(shape=[1], dtype='float32')
input_length = Input(shape=[1], dtype='int64')
label_length = Input(shape=[1], dtype='int64')

loss_output = Lambda(ctc_lambda)([pred_y, labels, input_length, label_length])

model = Model(inputs=input_data, outputs=loss_output)
