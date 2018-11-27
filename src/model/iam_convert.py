import numpy as np
import sys
from os import makedirs
from os.path import exists, join

from constants import *
from iam_reader import IamReader
from image_util import img_preprocess_nn, img_save


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
        # Ignore words with reserved Windows path characters.
        reserved = set(["<", ">", ":", "/", "\\", "|", "?", "*"])
        word_set = set(list(data.word))
        if not reserved.isdisjoint(word_set):
            continue

        img = img_preprocess_nn(img, data)

        fn = "{}-{}.png".format(str(i).zfill(6), data.word)
        img_save(join(dst, fn), img)


if __name__ == "__main__":
    convert()
