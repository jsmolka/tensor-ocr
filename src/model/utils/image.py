import cv2
import numpy as np

from model.alphabet import *
from model.constants import *


def load_img(fpath):
    """Loads a grayscale image from a path."""
    return cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)


def save_img(fpath, img):
    """Saves an image."""
    cv2.imwrite(fpath, img)


def resize(img, width, height):
    """Resizes an image using cubic iterpolation."""
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def resize_with_borders(img, width, height, fill=[255, 255, 255]):
    """Resizes an image by expanding it with borders."""
    h, w = img.shape
    bw = (width - w) // 2
    bh = (height - h) // 2

    return cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=fill)


def scale(img, factor):
    """
    Scales an image by a factor. The optimal interpolation gets chosen
    automatically depending on the factor.
    """
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


def scale_to_word_width(img, word, limit):
    """
    Resizes an image to its average word width. The limit can be used to 
    prevent the image from getting too big.
    """
    width = min(limit, len(word) * average_char_width)

    return scale_to_width(img, width)


def network_format(img):
    """
    Converts an image into the NN format. This includes transposing the image
    and converting it into float values between 0 and 1.
    """
    img = img.T
    img = img.astype("float32")
    img /= 255

    return img.reshape(input_shape)


def network_color(img):
    """
    Improves the image color for the NN. It eliminates lightgray areas by
    replacing them with and it darkens other colors.
    """

    def adjust_pixel(x):
        """Adjusts the color of a pixel."""
        if x <= 200:
            x = max(0, x - 32)
            return min(x, 128)
        else:
            return 255

    shape = img.shape
    
    img = img.reshape(-1)
    img = np.fromiter((adjust_pixel(x) for x in img), dtype=img.dtype, count=img.shape[0])

    return img.reshape(shape)


def network_preprocess(img, word=None):
    """
    Preprocesses an image for the NN. This includes using most of the above
    declared functions. If the word is known (like in training data), it can
    be passed to improve the preprocessing.
    """
    if word is not None: 
        if word not in set(list(alphabet_other)):
            img = scale_to_word_width(img, word, img_w)

    if img.shape[1] > img_w:
        img = scale_to_width(img, img_w)
    if img.shape[0] > img_h:
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
    """Loads an image and applies necessary NN preprocessing."""
    img = load_img(fpath)
    img = network_preprocess(img)

    return network_format(img)


def rotate_by_angle(img, angle, fill=[255, 255, 255]):
    """Rotates an image by an angle."""
    h, w = img.shape
    center = (w / 2, h / 2)
    rotate = cv2.getRotationMatrix2D(center, angle, 1.0)

    return cv2.warpAffine(img, rotate, (w, h), flags=cv2.INTER_LANCZOS4, borderValue=fill)
