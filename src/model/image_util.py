import cv2
import numpy as np

from constants import *


def resize_img(img, width, height, use_borders=False):
    """Resizes an image using iterpolation."""
    if use_borders:
        bw = round((width - img.shape[1]) / 2)
        bh = round((height - img.shape[0]) / 2)
        return cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=[255, 255, 255])
    else:
        return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def preprocess_img(img):
    """Removes background noise from an image and darkens the text color."""
    shape = img.shape

    img = img.reshape(-1)
    preprocess = lambda x: max(0, x - 32) if x <= 192 else 255
    img = np.fromiter((preprocess(x) for x in img), img.dtype, count=img.shape[0])

    return img.reshape(shape)


def load_img(path, resize=True):
    """Loads a grayscale image from a path."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if resize and img.shape != (img_h, img_w):
        img = resize_img(img, img_w, img_h)

    return img


def load_nn_img(path, preprocess=True):
    """Loads a image from a path and converts it into the NN format."""
    img = load_img(path)

    if preprocess:
        img = preprocess_img(img)
    img = img.T
    img = img.astype("float32")
    img /= 255

    return img.reshape(input_shape)


def save_img(path, img):
    """Saves an image."""
    cv2.imwrite(path, img)
