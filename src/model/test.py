import sys
import numpy as np
from glob import iglob
from os.path import join

from model.common import input_dir, file_to_word
from model.interface import predict_word


def test():
    """Tests the neural network."""
    src = input_dir("Converted IAM dataset")
    amount = int(input("Amount of test cases: "))

    correct = 0
    pattern = join(src, "*.png")
    for i, path in enumerate(iglob(pattern), start=1):
        word = file_to_word(path)
        pred = predict_word(path)

        if word == pred:
            correct += 1

        print("{} / {} / {} | {} | {} -> {}".format(
            str(i).rjust(3),
            str(correct).rjust(3),
            str(amount).rjust(3),
            str(word == pred).ljust(5),
            word.ljust(15),
            pred
        ))

        if i == amount:
            break
