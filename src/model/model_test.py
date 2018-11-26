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

    amount = int(input("Test case amount: "))

    correct = 0
    for i, fl in enumerate(iglob(join(src, "*.png")), start=1):
        word = basename(fl)[7:-4]
        words = probable_words(fl)

        if word in words:
            correct += 1

        print("{} / {} / {} | {} | {} -> {}".format(
            str(i).rjust(3),
            str(correct).rjust(3),
            str(amount).rjust(3),
            str(word in words).ljust(5),
            word.ljust(12),
            words
        ))

        if i == amount:
            break


if __name__ == "__main__":
    test()
