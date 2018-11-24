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

    img = img.astype('float32')
    img /= 255

    return img
    

def main():
    """Main function."""
    model = load_model("model.json", "weights.h5")

    img = load_img(r"C:\Users\Julian\Desktop\parsed_data\000002-MOVE.png")
    
    # Todo: support other keras channel
    data = np.array(img)
    data = np.reshape(data, (1, 128, 64, 1))
    
    labels = np.ones(shape=(80, 32))
    input_length = np.ones(shape=(1, 1))
    label_length = np.ones(shape=(1, 1))

    model.outputs = [model.get_layer("predict_y").output]  # Softmax layer output
    pred = model.predict([data, labels, input_length, label_length])

    print("Prediction shape:", pred.shape)
    print(pred.shape)
    pred = pred[0]
    for i in range(pred.shape[0]):
        # index = np.argmax(pred[i])
        print(pred[i][-1])



if __name__ == "__main__":
    main()
