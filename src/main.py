import numpy as np
import sys
import random

from model.iam.analyse import analyse
from model.iam.convert import convert
from model.iam.rotate import rotate
from model.train import train
from model.test import test


def main(argv):
    """Main function."""
    random.seed(100)
    np.random.seed(100)

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
