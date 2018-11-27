import numpy as np
import keras.backend as K
from keras.models import model_from_json
from keras.optimizers import SGD

from data.dataprovider import data_path
from model.constants import *
from model.image_util import load_network_img

_model = None


def load_model(json_file, weights_file):
    """Loads the model from json and weight files."""
    with open(json_file, "r") as infile:
        model = model_from_json(infile.read())
    
    model.load_weights(weights_file)

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])
    
    return model


def init_model():
    """Initializes the model."""
    global _model

    _model = load_model(
        data_path("model.json"),
        data_path("weights.h5")
    )


def get_model():
    """Returns the loaded model."""
    global _model

    if _model is None:
        init_model()

    return _model


def decode_ctc(prediction, top_paths):
    """Decodes the CTC output."""
    beam_width = max(5, top_paths)

    decoded = K.ctc_decode(
        prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
        greedy=False, beam_width=beam_width, top_paths=top_paths
    )[0]

    return [K.get_value(decoded[i])[0] for i in range(top_paths)]


def decode_label(label):
    """Decodes the label into a string."""
    return "".join([alphabet[index] for index in label if index < alphabet_size])


def decode_prediction(prediction, top_paths):
    """Decodes a model prediction."""
    labels = decode_ctc(prediction, top_paths)

    return [decode_label(label) for label in labels]


def predict(img):
    """Returns the prediction for an image."""
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)

    return get_model().predict(img)


def probable_words(img, count=10):
    """Gets the most probable words for an image (path)."""
    if isinstance(img, str):
        img = load_network_img(img)

    prediction = predict(img) 
    return decode_prediction(prediction, count)
