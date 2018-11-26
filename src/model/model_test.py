import sys
import numpy as np
from glob import iglob
from os.path import basename, exists, join

from model_interface import probable_words


def test():
    """Tests the neural network."""
    src = input("IAM data dir: ")
    if not exists(src):
        print("IAM data dir does not exist.")
        sys.exit(1)

    limit = int(input("Test data limit: "))

    for i, fl in enumerate(iglob(join(src, "*.png")), start=1):
        word = basename(fl)[7:-4]
        words = probable_words(fl)

        print(word, words, word in words)

        if i == limit:
            break


if __name__ == "__main__":
    test()
