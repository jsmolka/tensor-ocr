import sys
from os import makedirs
from os.path import basename, exists


def input_dir(name, create=False):
    """Asks for a directory."""
    src = input("{} directory: ".format(name))
    if not exists(src):
        if create:
            makedirs(src)
        else:
            print("Directory does not exist.")
            sys.exit()

    return src


def file_to_word(path):
    """Returns the word of a file."""
    return basename(path)[7:-4]


def word_to_file(index, word):
    """Creates a filename from an index and a word."""
    return "{}-{}.png".format(str(index).zfill(6), word)
