import numpy as np
import os
import cv2
import keras
from keras.models import Sequential, Model
from keras.layers import (Conv2D, MaxPooling2D, LSTM, Flatten, 
TimeDistributed, Activation, Dense, Input, Lambda, Bidirectional, 
Dropout, Reshape, GRU)
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD
from keras import backend as K
from os.path import join, split, splitext

train_size = 2000
validate_size = 100
cnn_size = 16
rnn_size = 512
dense_size = 32
kernel_size = (3, 3)
pool_size = 2

batch_size = 128
epochs = 5
rows, cols = 128, 64
channels = 1
alphabet_size = 78
label_max_length = 32

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
    predict_y, labels, input_length, label_length = args
    predict_y = predict_y[:, 2:, :]
    return K.ctc_batch_cost(labels, predict_y, input_length, label_length)

src = input("Source dir: ")
files = os.listdir(src)
file_iter = iter(files)

validate_x, validate_y = load_iam_data(src, file_iter, validate_size)
train_x, train_y = load_iam_data(src, file_iter, train_size)

train_x_input_length = np.full(shape=(train_size), fill_value=30, dtype=int)

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

train_y_length = np.ndarray(shape=(train_size))
for i, item in enumerate(train_y):
    train_y_length[i] = len(item)

print(input_shape)
input_data = Input(name="input_data", shape=input_shape, dtype='float32')
model = Conv2D(cnn_size, kernel_size=kernel_size, activation='relu', kernel_initializer='he_normal')(input_data)
model = MaxPooling2D(pool_size=(pool_size, pool_size))(model)
model = Conv2D(cnn_size, kernel_size=kernel_size, activation='relu', kernel_initializer='he_normal')(model)
model = MaxPooling2D(pool_size=(pool_size, pool_size))(model)

convert_cnn_to_rnn = (32, 210)
model = Reshape(target_shape=convert_cnn_to_rnn)(model)
model = Dense(dense_size, activation='relu', name='dense1')(model)

gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(model)
gru_1b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', go_backwards=True, name='gru1b')(model)
gru1_merged = add([gru_1, gru_1b])
gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', go_backwards=True, name='gru2b')(gru1_merged)

model = Dense(alphabet_size+1, kernel_initializer='he_normal', name='dense2')(concatenate([gru_2, gru_2b]))
predict_y = Activation('softmax', name='predict_y')(model)

labels = Input(shape=[label_max_length], dtype='float32', name='labels')
input_length = Input(shape=[1], dtype='int64', name='input_length')
label_length = Input(shape=[1], dtype='int64', name='label_length')

loss_output = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([predict_y, labels, input_length, label_length])

model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_output)

model.compile(loss={'ctc': lambda train_y, predict_y: predict_y}, optimizer='sgd')
model.summary()

output = np.ndarray(shape=train_size)

print(train_x.ndim)
print(train_y.ndim)
print(train_x_input_length.ndim)
print(train_y_length.ndim)

print(train_x.shape)
print(train_y.shape)
print(train_x_input_length.shape)
print(train_y_length.shape)

model.fit([train_x, train_y, train_x_input_length, train_y_length], output)

