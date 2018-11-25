import cv2
import keras
import numpy as np
from glob import iglob
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation, Conv2D, Dense, GRU, Input, Lambda, MaxPooling2D, Reshape
from keras.layers.merge import add, concatenate
from keras.optimizers import SGD
from keras.utils import plot_model
from os.path import join, basename

img_w = 128
img_h = 64

train_size = 2000
valid_size = 200
conv_size = 16
rnn_size = 512
dense_size = 32
kernel_size = (3, 3)
pool_size = 2

batch_size = 128
epochs = 1
channels = 1

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.!?,:;+-*()#&/'\" "
alphabet_size = len(alphabet)
label_max_length = 32


def encode_label(string):
    """
    Encodes a string into a list of its character alphabet positions.
    "abcdef" -> [0, 1, 2, 3, 4, 5]
    """
    return [alphabet.index(char) for char in string]


def load_iam_data(globber, size):
    """Parses image data into numpy arrays."""
    x = np.ndarray(shape=(size, img_w, img_h))
    y = np.full(shape=(size, label_max_length), fill_value=alphabet_size)

    for i in range(size):
        path = next(globber)
        word = str(basename(path))[7:-4]
        if len(word) <= label_max_length:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            x[i] = cv2.resize(img, (img_w, img_h)).T
            y[i, 0:len(word)] = encode_label(word)

    return x, y


def ctc_lambda(args):
    """Lambda for CTC loss."""
    y_pred, labels, input_length, label_length = args
    # y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def train():
    """Trains and saves the model."""
    # Load and shape the training and validation data
    globber = iglob(join(input("IAM data dir: "), "*.png"))
    x_train, y_train = load_iam_data(globber, train_size)
    x_valid, y_valid = load_iam_data(globber, valid_size)

    x_train = x_train.astype("float32") / 255
    x_valid = x_valid.astype("float32") / 255

    if K.image_data_format() == "channels_first":
        input_shape = (channels, img_w, img_h)
    else:
        input_shape = (img_w, img_h, channels)

    reshape = lambda array: array.reshape(array.shape[0], *input_shape)
    x_train = np.expand_dims(x_train, axis=3)
    x_valid = np.expand_dims(x_valid, axis=3)

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

    # Create the input layer
    labels = Input(shape=(label_max_length,), dtype="float32", name="labels")
    input_length = Input(shape=(1,), dtype="int64", name="input_length")
    label_length = Input(shape=(1,), dtype="int64", name="label_length")

    loss_output = Lambda(
        ctc_lambda, output_shape=(1,), 
        name="ctc")([y_pred, labels, input_length, label_length]
    )

    # Compile model
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model = Model(inputs=[input_data, labels, input_length, label_length], outputs=loss_output)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=sgd)

    output = np.ndarray(shape=(train_size,))

    model.fit(
        x=[x_train, y_train, input_length_x, label_length_y], 
        y=output, 
        # batch_size=batch_size
        epochs=epochs,
        verbose=1,
        validation_data=([x_valid, y_valid, input_length_x_valid, label_length_y_valid], y_valid)
    )

    # Save model
    # with open("model.json", "w") as json_file:
    #     json_file.write(model.to_json())
    # model.save_weights("weights.h5")

    
    ##########################
    # Test the created model #
    ##########################

    test_cases = 5

    # Remove the extra input layers from the model and transfer the weights
    stripped = Model(inputs=input_data, outputs=y_pred)
    for idx, layer in enumerate(model.layers[:-4]):
        stripped.layers[idx].set_weights(layer.get_weights())

    stripped_data = []
    for t in x_train[:test_cases]:
        pred = stripped.predict(np.array([t]))
        word = [np.argmax(x) for x in pred[0]]
        stripped_data.append(word)
        print(word)

    # Compare to original model
    original_data = []
    for t in x_train[:test_cases]:
        model.outputs = [model.get_layer("y_pred").output]
        pred = model.predict([np.array([t]), y_train, input_length_x, label_length_y])
        word = [np.argmax(x) for x in pred[0]]
        original_data.append(word)

    if stripped_data != original_data:
        print("Data between stripped and original model differs.")
        for data in original_data:
            print(data)


if __name__ == "__main__":
    train()
