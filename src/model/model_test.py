import sys
import numpy as np
from glob import iglob
from os.path import basename, exists, join

from model.common import input_dir, file_word
from model.model_interface import probable_words


def test():
    """Tests the neural network."""
    src = input_dir("Converted IAM dataset")
    amount = int(input("Amount of test cases: "))

    correct = 0
    for i, fpath in enumerate(iglob(join(src, "*.png")), start=1):
        word = file_word(fpath)
        words = probable_words(fpath, count=5)

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
