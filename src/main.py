import sys

from model.iam.analyse import analyse
from model.iam.convert import convert
from model.train import train
from model.test import test


def main(argv):
    """Main function."""
    args = argv[1:]

    arg_analyse = "--analyse"
    arg_convert = "--convert"
    arg_train = "--train"
    arg_test = "--test"

    if len(args) == 0:
        print("Please use one or more of the following arguments.")
        print(arg_analyse)
        print(arg_convert)
        print(arg_train)
        print(arg_test)
    
    funcs = {
        arg_analyse: analyse,
        arg_convert: convert,
        arg_train: train,
        arg_test: test
    }

    for arg in args:
        if arg in funcs:
            funcs[arg]()
        else:
            print("Invalid argument \"{}\".".format(arg))


if __name__ == "__main__":
    main(sys.argv)
