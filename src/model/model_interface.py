import cv2
import numpy as np
import keras.backend as K
from keras.models import model_from_json
from keras.optimizers import SGD

img_w = 128
img_h = 64

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.!?,:;+-*()#&/'\" "
alphabet_size = len(alphabet)


def load_model(json, weights):
    """Loads the model from json and weight files."""
    model = None
    with open(json, "r") as json_file:
        model = model_from_json(json_file.read())
    
    sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)

    model.load_weights(weights)
    model.compile(loss="binary_crossentropy", optimizer=sgd, metrics=['accuracy'])
    
    return model


def load_img(path):
    """Loads data from a path."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_w, img_h))

    img = img.astype('float32')
    img /= 255
    img = np.expand_dims(img.T, axis=2)

    return img
    

def main():
    """Main function."""
    model = load_model("model.json", "weights.h5")

    img = load_img(r"C:\Users\Julian\Desktop\parsed_data\000215-that.png")
    
    # Todo: support other keras channel, like in model
    data = np.array([img])
    # data = np.reshape(data, (1, img_w, img_h, 1))
    print(data.shape)

    pred = model.predict(data)

    print("Prediction shape:", pred.shape)
    print(pred.shape)
    pred = pred[0]
    for i in range(pred.shape[0]):
        index = np.argmax(pred[i])
        if index < alphabet_size:
            print(alphabet[index])



if __name__ == "__main__":
    main()
