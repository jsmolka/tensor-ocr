import cv2
import numpy as np

from model.alphabet import *
from model.constants import *


def load_img(fpath):
    """Loads an image from a path."""
    return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)


def save_img(fpath, img):
    """Saves an image."""
    cv2.imwrite(fpath, img)


def resize(img, width, height):
    """Resizes an image using iterpolation."""
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def resize_with_borders(img, width, height):
    """Resizes an image using borders."""
    h, w = img.shape
    bw = (width - w) // 2
    bh = (height - h) // 2

    return cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def scale(img, factor):
    """Scales an image by a factor."""
    interpolation = cv2.INTER_AREA if factor <= 1 else cv2.INTER_LANCZOS4

    return cv2.resize(img, dsize=(0, 0), fx=factor, fy=factor, interpolation=interpolation)


def scale_to_width(img, width):
    """Scales an image to a certain width."""
    _, w = img.shape

    return scale(img, width / w)


def scale_to_height(img, height):
    """Resizes an image to a certain height."""
    h, _ = img.shape

    return scale(img, height / h)


def scale_to_word_width(img, word, limit=img_w):
    """Resizes an image to the average word width."""
    width = len(word) * average_char_width
    width = min(width, limit)

    return scale_to_width(img, width)


def network_format(img):
    """Converts an image into the NN format."""
    img = img.T
    img = img.astype("float32")
    img /= 255

    return img.reshape(input_shape)


def network_color(img):
    """Adjusts the color of an image for the NN."""
    shape = img.shape
    
    img = img.reshape(-1)
    adjust = lambda x: min(max(0, x - 32), 128) if x <= 200 else 255
    img = np.fromiter((adjust(x) for x in img), img.dtype, count=img.shape[0])

    return img.reshape(shape)


def network_preprocess(img, word=None):
    """Preprocesses an image for the NN."""
    if word is not None and word not in set(alphabet_other):
        img = scale_to_word_width(img, word)

    _, w = img.shape
    if w > img_w:
        img = scale_to_width(img, img_w)
    
    h, _ = img.shape
    if h > img_h:
        img = scale_to_height(img, img_h)

    img = network_color(img)
    
    img = resize_with_borders(img, img_w, img_h)
    if img.shape != (img_h, img_w):
        img = resize(img, img_w, img_h)

    return img


def load_training_img(fpath):
    """Loads a training image."""
    return network_format(load_img(fpath))


def load_network_img(fpath):
    """Loads an image and converts it for the NN."""
    img = load_img(fpath)
    img = network_preprocess(img)
    save_img("C:/Users/Julian/Desktop/test.png", img)

    return network_format(img)


def rotate_img(img, angle):
    """Rotates an image by an angle."""
    center = tuple(np.array(img.shape[1::-1]) / 2)
    rotate = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(img, rotate, img.shape[1::-1], flags=cv2.INTER_LINEAR, borderValue=[255, 255, 255])
