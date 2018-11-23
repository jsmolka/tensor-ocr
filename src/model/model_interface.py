import cv2
import numpy as np
from keras.models import model_from_json


def load_model(json, weights):
    """Loads the model from json and weight files."""
    model = None
    with open(json, "r") as json_file:
        model = model_from_json(json_file.read())
    model.load_weights(weights)
    model.compile(loss={'ctc': lambda train_y, predict_y: predict_y}, optimizer='sgd', metrics=['accuracy'])
    
    return model


def load_img(path):
    """Loads data from a path."""
    rows = 128
    cols = 64

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (rows, cols))

    return img
    

def main():
    """Main function."""
    model = load_model("model.json", "weights.h5")

    img = load_img(r"C:\Users\Julian\Desktop\parsed_data\000002-MOVE.png")
    data = np.array([img], dtype=np.uint8)
    data = data.reshape(128, 64, 1)

    data = data.astype('float32')
    data /= 255
    
    print("data", data.shape)

    pred = model.predict(data, batch_size=128)
    print("pred", pred)


if __name__ == "__main__":
    main()
