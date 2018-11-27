import cv2
import numpy as np

from constants import *


def img_resize(img, width, height):
    """Resizes an image using iterpolation."""
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def img_resize_with_borders(img, width, height, fill=[255, 255, 255]):
    """Resizes an image using borders."""
    h, w = img.shape
    bw = (width - w) // 2
    bh = (height - h) // 2
    img = cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=fill)

    if img.shape != (height, width):
        img = img_resize(img, width, height)

    return img


def img_scale_to_width(img, width):
    """Scales an image to a certain width."""
    _, w = img.shape
    fx = width / w

    return cv2.resize(img, dsize=(0, 0), fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)


def img_scale_to_height(img, height):
    """Resizes an image to a certain height."""
    h, _ = img.shape
    fy = height / h

    return cv2.resize(img, dsize=(0, 0), fx=fy, fy=fy, interpolation=cv2.INTER_CUBIC)


def img_adjust(img):
    """Adjusts an image for the NN."""

    def adjust(x):
        """Adjusts a single value."""
        if x <= 192:
            # Darken the text in general.
            x = max(0, x - 32)
            # Set a minimal text darkness.
            return min(x, 112)
        else:
            # Remove lightgray background noise.
            return 255
    
    shape = img.shape
    
    # Flatten the image without creating a copy.
    img = img.reshape(-1)
    img = np.fromiter((adjust(x) for x in img), img.dtype, count=img.shape[0])

    return img.reshape(shape)


def img_load(path):
    """Loads an image from a path."""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def img_preprocess_nn(img, data=None):
    """Preprocesses an image for the NN."""
    adjusted = False
    h, w = img.shape
    # Adjust the image while it is still small.
    if w < img_w and h < img_h:
        img = img_adjust(img)
        adjusted = True

    # Properly scale the image to its average word width. Limit it to the
    # desired image width to prevent multiple scaling operations. Ignore other 
    # characters because they are way smaller.
    if data is not None:
        if not data.word in set(alphabet_other):
            width = len(data.word) * average_char_width
            width = min(width, img_w)
            img = img_scale_to_width(img, width)

    # Make sure the the image has the desired width.
    _, w = img.shape
    if w > img_w:
        img = img_scale_to_width(img, img_w)
    
    # Make sure that the image has the desired height.
    h, _ = img.shape
    if h > img_h:
        img = img_scale_to_height(img, img_h)

    # Adjust the image before adding borders.
    if not adjusted:
        img = img_adjust(img)

    # Add borders to achieve the desired with.
    return img_resize_with_borders(img, img_w, img_h)


def img_load_nn(path, training=False):
    """Loads an image from a path and converts it into the NN format."""
    img = img_load(path)

    # Assume that training data already has the desired format.
    if not training:
        img = img_convert_nn(img)

    img = img.T
    img = img.astype("float32")
    img /= 255

    return img.reshape(input_shape)


def img_save(path, img):
    """Saves an image."""
    cv2.imwrite(path, img)
