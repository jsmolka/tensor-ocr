import keras
import numpy as np
import sys
from glob import iglob
from keras import backend as K
from keras.layers import Activation, Conv2D, Dense, GRU, Input, Lambda, MaxPooling2D, Reshape
from keras.layers.merge import add, concatenate
from keras.models import Sequential, Model
from keras.optimizers import SGD
from os.path import join, basename, exists

from constants import *
from image_util import load_nn_img

dataset_size = 113000
valid_ratio = 0.2
valid_size = round(dataset_size * valid_ratio)
train_size = dataset_size - valid_size

kernel_size = (3, 3)
conv_size = 16
pool_size = 2
dense_size = 32
rnn_size = 512

batch_size = 128
epochs = 150

max_label_length = 32


def encode_string(string):
    """
    Encodes a string into a list of its character alphabet positions.
    "abcdef" -> [0, 1, 2, 3, 4, 5]
    """
    return [alphabet.find(char) for char in string]


def load_data(globber, size):
    """Parses image data into numpy arrays."""
    x = np.ndarray(shape=(size, *input_shape), dtype=np.float32)
    y = np.full(shape=(size, max_label_length), fill_value=alphabet_size)

    for i in range(size):
        try:
            path = next(globber)
        except StopIteration:
            print("Tried loading more files than exist")
            sys.exit(1)

        word = basename(path)[7:-4]
        word_size = len(word)

        if word_size <= max_label_length:
            x[i] = load_nn_img(path, preprocess=False)
            y[i, :word_size] = encode_string(word)

    return x, y


def convert_model(model, inputs, outputs):
    """Converts the trained model into one without the extra input layers."""
    new_model = Model(inputs=inputs, outputs=outputs)

    # The last four layers are only needed for the loss calculation and can be
    # removed once the model is fully trained.
    for i, layer in enumerate(model.layers[:-4]):
        new_model.layers[i].set_weights(layer.get_weights())

    return new_model


def save_model(model, json_path, h5_path):
    """Saves a model."""
    with open(json_path, "w") as json_file:
        json_file.write(model.to_json())

    model.save_weights(h5_path)


def train():
    """Trains and saves the model."""
    src = input("IAM data dir: ")
    if not exists(src):
        print("IAM data dir does not exist.")
        sys.exit(1)

    # Load training and validation data
    globber = iglob(join(src, "*.png"))
    x_train, y_train = load_data(globber, train_size)
    x_valid, y_valid = load_data(globber, valid_size)

    input_length_x = np.full(shape=(train_size,), fill_value=32, dtype=int)
    label_length_y = np.ndarray(shape=(train_size,))

    input_length_x_valid = np.full(shape=(valid_size,), fill_value=32, dtype=int)
    label_length_y_valid = np.ndarray(shape=(valid_size,))

    for i, item in enumerate(y_train):
        label_length_y[i] = item.shape[0]

    for i, item in enumerate(y_valid):
        label_length_y_valid[i] = item.shape[0]

    # Create the model using different layers
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

    gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer="he_normal", name="gru1")(inner)
    gru_1b = GRU(rnn_size, return_sequences=True, kernel_initializer="he_normal", go_backwards=True, name="gru1b")(inner)
    gru1_merged = add([gru_1, gru_1b])
    
    gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer="he_normal", name="gru2")(gru1_merged)
    gru_2b = GRU(rnn_size, return_sequences=True, kernel_initializer="he_normal", go_backwards=True, name="gru2b")(gru1_merged)
    gru2_concat = concatenate([gru_2, gru_2b])

    inner = Dense(alphabet_size + 1, kernel_initializer="he_normal", name="dense2")(gru2_concat)
    y_pred = Activation("softmax", name="y_pred")(inner)

    Model(inputs=input_data, outputs=y_pred).summary()

    # Add additional input layers for the loss function
    labels = Input(shape=(max_label_length,), dtype="float32", name="labels")
    input_length = Input(shape=(1,), dtype="int64", name="input_length")
    label_length = Input(shape=(1,), dtype="int64", name="label_length")

    loss_output = Lambda(
        lambda args: K.ctc_batch_cost(*args), output_shape=(1,), 
        name="ctc")([labels, y_pred, input_length, label_length]
    )

    # Compile model
    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_output)
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=sgd, metrics=["accuracy"])

    output = np.ndarray(shape=(train_size,))

    model.fit(
        x=[x_train, y_train, input_length_x, label_length_y], 
        y=output, 
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=([x_valid, y_valid, input_length_x_valid, label_length_y_valid], y_valid)
    )

    model = convert_model(model, inputs=input_data, outputs=y_pred)
    save_model(model, "model.json", "weights.h5")


if __name__ == "__main__":
    train()
