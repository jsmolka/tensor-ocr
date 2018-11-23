import numpy as np
import os
import cv2
import keras
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Input
from keras.layers import Lambda, GRU, Reshape
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD
from keras import backend as K
from os.path import join

train_size = 108000
validate_size = 5000
cnn_size = 16
rnn_size = 512
dense_size = 32
kernel_size = (3, 3)
pool_size = 2

batch_size = 128
epochs = 1
rows, cols = 128, 64
channels = 1
alphabet = (" !\"#&'()*+,-./0123456789:;?"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "abcdefghijklmnopqrstuvwxyz")
alphabet_size = len(alphabet)
label_max_length = 32


def load_iam_data(path, file_iter, size):
    """Parses images into numpy arrays."""
    x = np.ndarray(shape=(size, rows, cols))
    y = np.ones(shape=[size, label_max_length]) * alphabet_size

    for i in range(0, size):
        fi = next(file_iter)
        word = str(fi)[7:-4]
        if len(word) >= 32:
            continue

        tmp = cv2.imread(join(path, fi), cv2.IMREAD_GRAYSCALE)
        x[i] = tmp.reshape(rows, cols)
        y[i, 0:len(word)] = encode_label(word)
    return x, y


def ctc_lambda(args):
    predict_y, labels, input_length, label_length = args
    # predict_y = predict_y[:, 2:, :]
    return K.ctc_batch_cost(labels, predict_y, input_length, label_length)


def encode_label(word):
    """Encodes a string into an array of it's characters positions in the alphabet."""
    onehot = []
    for i in range(0, len(word)):
        for j in range(0, alphabet_size):
            if word[i] == alphabet[j]:
                onehot.append(j)
                break
    return onehot


def decode_label(onehot):
    """Decodes an array of character positions into a string."""
    word = []
    for i in range(0, len(onehot)):
        if onehot[i] == alphabet_size:  # CTC blank character
            word.append("")
        else:
            word.append(alphabet[onehot[i]])
    word = ''.join(word)
    return word

src = input("Source dir: ")
files = os.listdir(src)
file_iter = iter(files)

# validation data
validate_x, validate_y = load_iam_data(src, file_iter, validate_size)
# training data
train_x, train_y = load_iam_data(src, file_iter, train_size)

input_length_x = np.full(shape=(train_size), fill_value=32, dtype=int)
label_length_y = np.ndarray(shape=(train_size))

input_length_validate_x = np.full(shape=(validate_size), fill_value=32, dtype=int)
label_length_validate_y = np.ndarray(shape=(validate_size))

for i, item in enumerate(train_y):
    label_length_y[i] = len(item)

for i, item in enumerate(validate_y):
    label_length_validate_y[i] = len(item)

if K.image_data_format() == 'channels_first':
    validate_x = validate_x.reshape(validate_x.shape[0], channels, rows, cols)
    train_x = train_x.reshape(train_x.shape[0], channels, rows, cols)
    input_shape = (channels, rows, cols)
else:
    validate_x = validate_x.reshape(validate_x.shape[0], rows, cols, channels)
    train_x = train_x.reshape(train_x.shape[0], rows, cols, channels)
    input_shape = (rows, cols, channels)

# cast to intervall 0..1
validate_x = validate_x.astype('float32')
validate_x /= 255
train_x = train_x.astype('float32')
train_x /= 255

# create model with multiple inputs
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

# the encoded strings: train_y
labels = Input(shape=[label_max_length], dtype='float32', name='labels')
# input sequence length for CTC -> output sequence length of the previous layer
input_length = Input(shape=[1], dtype='int64', name='input_length')
# length of labels including whitespace[0, 2, 3, 20] -> 4
label_length = Input(shape=[1], dtype='int64', name='label_length')

# CTC loss function
loss_output = Lambda(ctc_lambda, output_shape=(1,), name='ctc')([predict_y, labels, input_length, label_length])

# specify input and output data
model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_output)

# specify loss function(CTC) and optimizer
model.compile(loss={'ctc': lambda train_y, predict_y: predict_y}, optimizer='sgd', metrics=['accuracy'])
model.summary()

output = np.ndarray(shape=train_size)
validate_output = np.ndarray(shape=validate_size)

# train model
model.fit([train_x, train_y, input_length_x, label_length_y], output, 
          epochs=epochs,
          verbose=1,
          validation_data=([validate_x, validate_y, input_length_validate_x, label_length_validate_y], validate_y))

# Save model
with open("model.json", "w") as json_file:
    json_file.write(model.to_json())

model.save_weights("weights.h5")
