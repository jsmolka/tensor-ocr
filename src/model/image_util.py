import cv2

from constants import *


def resize(img, width, height):
    """Resizes an image using iterpolation."""
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def resize_using_border(img, width, height):
    """Resizes an image by adding a white border."""
    h, w = img.shape
    bw = round((width - w) / 2)
    bh = round((height - h) / 2)

    return cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def threshold(img):
    """Removes lightgray background noise from an image."""
    img[img > 192] = 255

    return img

def load_img(path, auto_resize=True):
    """Loads a grayscale image from a path."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if auto_resize and img.shape != (img_h, img_w):
        img = resize(img, img_w, img_h)

    return img


def load_nn_img(path):
    """Loads a image from a path and converts it into the NN format."""
    img = load_img(path)

    img = threshold(img)
    img = img.T
    img = img.astype("float32")
    img /= 255

    return img.reshape(input_shape)


def save_img(path, img):
    """Saves an image."""
    cv2.imwrite(path, img)
