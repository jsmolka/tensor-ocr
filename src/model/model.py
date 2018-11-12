import numpy as np
import os
import cv2
import keras
from keras.models import Sequential
from keras.layers import (Conv2D, MaxPooling2D, LSTM, Flatten, 
TimeDistributed, Activation, Dense, Input, Lambda)
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
        tmp = cv2.imread(join(path, fi))
        tmp = tmp.reshape(rows, cols, 3)

        for j, row in enumerate(tmp):
            for k, col in enumerate(row):
                x[i][j][k] = col[2]
        y[i] = str(fi)[7:-4]

    return x, y

def ctc_lambda(args):
    pred_y, labels, input_length, label_length = args
    pred_y = pred_y[:, 2:, :]
    return K.ctc_batch_cost(labels, pred_y, input_length, label_length)

src = input("Source dir: ")
files = os.listdir(src)
file_iter = iter(files)

validate_x, validate_y = load_iam_data(src, file_iter, 100)
train_x, train_y = load_iam_data(src, file_iter, 500)

if K.image_data_format() == 'channels_first':
    validate_x = validate_x.reshape(validate_x.shape[0], channels, rows, cols)
    train_x = train_x.reshape(train_x.shape[0], channels, rows, cols)
    input_shape = (channels, rows, cols)
else:
    validate_x = validate_x.reshape(validate_x.shape[0], rows, cols, channels)
    train_x = train_x.reshape(train_x.shape[0], rows, cols, channels)
    input_shape = (rows, cols, channels)

validate_x = validate_x.astype('float32')
validate_y = validate_y.astype('str')
validate_x /= 255

train_x = train_x.astype('float32')
train_y = train_y.astype('str')
train_x /= 255

cnn = Sequential()
cnn.add(Conv2D(64, kernel_size=(3,3), activation='relu', input_shape=input_shape))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
cnn.add(Conv2D(64, kernel_size=(2,2), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(64, kernel_size=(2,2), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(32))

model = Sequential()
model.add(TimeDistributed(cnn))
model.add(LSTM(64))
model.add(LSTM(64))
model.add(Dense(78, kernel_initializer='he_normal'))
model.add(Activation('softmax'))

labels = Input(shape=[32], dtype='float32')
input_length = Input(shape=[1], dtype='int64')
label_length = Input(shape=[1], dtype='int64')

model.add(Lambda(ctc_lambda, output_shape=(1,), arguments=[model, labels, input_length, label_length]))

model.compile(loss={}, optimizer='sgd', metrics=['accuracy'])

model.fit(train_x, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(validate_x, validate_y))

score = model.evaluate(validate_x, validate_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])