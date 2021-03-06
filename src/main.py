import numpy as np
import random
import sys
import tensorflow as tf

from model.iam.analyse import analyse
from model.iam.convert import convert
from model.iam.rotate import rotate
from model.train import train
from model.test import test


def seed(value):
    """Seeds random generators for reproducible results."""
    tf.set_random_seed(value)
    np.random.seed(value)
    random.seed(value)


def main(argv):
    """Main function."""
    seed(0)

    args = argv[1:]

    funcs = {
        "--analyse": analyse,
        "--convert": convert,
        "--rotate": rotate,
        "--train": train,
        "--test": test
    }

    if len(args) == 0:
        print("Please use one or more of the following arguments.")
        for key in funcs.keys():
            print(key)

    for arg in args:
        if arg in funcs:
            funcs[arg]()
        else:
            print("Invalid argument \"{}\".".format(arg))


if __name__ == "__main__":
    main(sys.argv)
