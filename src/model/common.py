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


def file_word(fpath):
    """Returns the word of a file."""
    return basename(fpath)[7:-4]
