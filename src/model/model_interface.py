import cv2
import numpy as np
import keras.backend as K
from keras.models import model_from_json
from keras.optimizers import SGD

from constants import *
from image_util import load_nn_img


def load_model(json_path, weights_path):
    """Loads the model from json and weight files."""
    with open(json_path, "r") as json_file:
        model = model_from_json(json_file.read())
    
    model.load_weights(weights_path)

    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])
    
    return model


_model = load_model("model.json", "weights.h5")


def decode_prediction(prediction, top_paths):
    """Decodes a model prediction."""
    results = []
    beam_width = max(5, top_paths)    
    for i in range(top_paths):
        labels = K.get_value(
            K.ctc_decode(
                prediction, input_length=np.ones(prediction.shape[0]) * prediction.shape[1],
                greedy=False, beam_width=beam_width, top_paths=top_paths
            )[0][i]
        )[0]

        chars = []
        for index in labels:
            if index < alphabet_size:
                chars.append(alphabet[index])

        results.append("".join(chars))

    return results


def model_prediction(img):
    """Returns the prediction for an image."""
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)

    return _model.predict(img)


def probable_words(img, count=5):
    """Gets the most probable words for an image (path)."""
    if isinstance(img, str):
        img = load_nn_img(img)

    prediction = model_prediction(img) 
    return decode_prediction(prediction, count)
