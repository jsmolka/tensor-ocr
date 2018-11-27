import numpy as np
import sys
from os import makedirs
from os.path import exists, join

from constants import *
from iam_reader import IamReader
from image_util import *


def convert():
    """Converts the given IAM data."""
    src = input("IAM source dir: ")
    if not exists(src):
        print("IAM source dir does not exist.")
        sys.exit(1)

    dst = input("Destination dir: ")
    if not exists(dst):
        makedirs(dst)

    reader = IamReader(src)
    for i, (data, img) in enumerate(reader.data_iter(), start=1):
        height, width = img.shape
        try:
            # Ignore words with reserved Windows path characters.
            reserved = set(["<", ">", ":", "/", "\\", "|", "?", "*"])
            word_set = set(list(data.word))
            if not reserved.isdisjoint(word_set):
                continue

            # Resize short words using borders to prevent them from getting too
            # big. They still need to be resized afterwards because they might
            # not have the desired size.
            if len(data.word) <= 2 and height < img_h and width < img_w:
                img = resize_img(img, img_w, img_h, use_borders=True)

            img = resize_img(img, img_w, img_h)
            img = preprocess_img(img)

            fn = "{}-{}.png".format(str(i).zfill(6), data.word)
            save_img(join(dst, fn), img)

        except Exception as e:
            print("Failed converting {} with error: {}".format(data.path, str(e)))


if __name__ == "__main__":
    convert()
