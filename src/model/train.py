import numpy as np
from glob import iglob
from keras import backend as K
from keras.layers import Activation, Conv2D, Dense, GRU, Input, Lambda, MaxPooling2D, Reshape, CuDNNGRU
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import SGD
from os.path import join
from random import shuffle

from data.dataprovider import data_path
from model.alphabet import *
from model.common import input_dir, file_to_word
from model.constants import *
from model.utils.image import load_training_img

dataset_size = 113#000
valid_ratio = 0.1
valid_size = round(dataset_size * valid_ratio)
train_size = dataset_size - valid_size

kernel_size = (3, 3)
conv_size = 16
pool_size = 2
dense_size = 32
rnn_size = 512

batch_size = 128
epochs = 200

max_label_length = 32


def encode_string(string):
    """
    Encodes a string into a list of its character alphabet positions.
    
    "abcdef" -> [0, 1, 2, 3, 4, 5]
    """
    return [alphabet.find(char) for char in string]


def load_paths(src, limit):
    """
    Loads file paths for a given source directory. Shuffling the files paths 
    prevents only using rotated images as validation data.
    """
    paths = []
    
    pattern = join(src, "*.png")
    for i, path in enumerate(iglob(pattern), start=1):
        paths.append(path)
        if i == limit:
            break

    shuffle(paths)
    return paths
        

def load_data(src):
    """Parses image data into numpy arrays."""
    x = np.ndarray(shape=(dataset_size, *input_shape), dtype=np.float32)
    y = np.full(shape=(dataset_size, max_label_length), fill_value=alphabet_size, dtype=np.uint8)
    z = np.ndarray(shape=(dataset_size,), dtype=np.uint8)

    paths = load_paths(src, dataset_size)
    for i, path in enumerate(paths):
        word = file_to_word(path)
        word_len = len(word)

        x[i] = load_training_img(path)
        y[i, :word_len] = encode_string(word)
        z[i] = word_len

    return x, y, z


def slice_data(x):
    """Slices data into training and validation data."""
    train = x[:train_size]
    valid = x[dataset_size - valid_size:]
    
    return train, valid


def convert_model(model, inputs, outputs):
    """Converts the trained model into one without the extra input layers."""
    new = Model(inputs=inputs, outputs=outputs)

    for i, layer in enumerate(model.layers[:-4]):
        new.layers[i].set_weights(layer.get_weights())

    return new


def save_model(model, json, weigts):
    """Saves a model."""
    with open(json, "w") as outfile:
        outfile.write(model.to_json())

    model.save_weights(weigts)


def train():
    """Trains and saves the model."""
    src = input_dir("Converted IAM dataset")
    x, y, z = load_data(src)

    x_train, x_valid = slice_data(x)
    y_train, y_valid = slice_data(y)
    y_train_label_length, y_valid_label_length = slice_data(z)

    x_train_input_length = np.full(shape=(train_size,), fill_value=max_label_length, dtype=np.uint8)
    x_valid_input_length = np.full(shape=(valid_size,), fill_value=max_label_length, dtype=np.uint8)

    input_data = Input(name="input_data", shape=input_shape, dtype="float32")
    inner = Conv2D(conv_size, kernel_size=kernel_size, padding="same", activation="relu", kernel_initializer="he_normal", name="conv1")(input_data)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name="max1")(inner)
    inner = Conv2D(conv_size, kernel_size=kernel_size, padding="same", activation="relu", kernel_initializer="he_normal", name="conv2")(inner)
    inner = MaxPooling2D(pool_size=(pool_size, pool_size), name="max2")(inner)

    conv_to_rnn_dims = (
        img_w // (pool_size ** 2),
        (img_h // (pool_size ** 2)) * conv_size
    )

    inner = Reshape(target_shape=conv_to_rnn_dims, name="reshape")(inner)
    inner = Dense(dense_size, activation="relu", name="dense1")(inner)

    if gpu_enabled:
        GRU_ = CuDNNGRU 
    else:
        GRU_ = GRU

    gru_1 = GRU_(rnn_size, return_sequences=True, kernel_initializer="he_normal", name="gru1")(inner)
    gru_1b = GRU_(rnn_size, return_sequences=True, kernel_initializer="he_normal", go_backwards=True, name="gru1b")(inner)
    gru1_merged = add([gru_1, gru_1b])
    
    gru_2 = GRU_(rnn_size, return_sequences=True, kernel_initializer="he_normal", name="gru2")(gru1_merged)
    gru_2b = GRU_(rnn_size, return_sequences=True, kernel_initializer="he_normal", go_backwards=True, name="gru2b")(gru1_merged)
    gru2_concat = concatenate([gru_2, gru_2b])

    inner = Dense(alphabet_size + 1, kernel_initializer="he_normal", name="dense2")(gru2_concat)
    y_pred = Activation("softmax", name="y_pred")(inner)

    Model(inputs=input_data, outputs=y_pred).summary()

    labels = Input(shape=(max_label_length,), dtype="float32", name="labels")
    input_length = Input(shape=(1,), dtype="int64", name="input_length")
    label_length = Input(shape=(1,), dtype="int64", name="label_length")

    loss_output = Lambda(
        lambda args: K.ctc_batch_cost(*args), output_shape=(1,), name="ctc"
    )([labels, y_pred, input_length, label_length])

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_output)
    model.compile(sgd, loss={"ctc": lambda y_true, y_pred: y_pred}, metrics=["accuracy"])

    model.fit(
        x=[x_train, y_train, x_train_input_length, y_train_label_length], 
        y=np.ndarray(shape=(train_size,)), 
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=([x_valid, y_valid, x_valid_input_length, y_valid_label_length], y_valid)
    )

    model = convert_model(model, inputs=input_data, outputs=y_pred)
    save_model(model, data_path("model.json"), data_path("weights.h5"))
