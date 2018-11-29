import sys
import numpy as np
from glob import iglob
from os.path import basename, exists, join

from model.common import input_dir, file_word
from model.interface import predict_word


def test():
    """Tests the neural network."""
    src = input_dir("Converted IAM dataset")
    amount = int(input("Amount of test cases: "))

    correct = 0
    for i, fpath in enumerate(iglob(join(src, "*.png")), start=1):
        word = file_word(fpath)
        prediction = predict_word(fpath)

        if word == prediction:
            correct += 1

        print("{} / {} / {} | {} | {} -> {}".format(
            str(i).rjust(3),
            str(correct).rjust(3),
            str(amount).rjust(3),
            str(word == prediction).ljust(5),
            word.ljust(15),
            prediction
        ))

        if i == amount:
            break
