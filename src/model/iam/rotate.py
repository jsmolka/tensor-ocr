from glob import glob
from os.path import join
from random import choice, randint

from model.common import input_dir, word_to_file, file_to_word
from model.utils.image import load_img, save_img, rotate_by_angle


def random_angle():
    """Returns an angle within the interval [-5, -1] or [1, 5]"""
    return choice([-1, 1]) * randint(1, 5)


def rotate():
    """Converts the IAM dataset."""
    src = input_dir("Converted IAM dataset")
    dst = input_dir("Destination", create=True)
    
    paths = glob(join(src, "*.png"))
    start = len(paths) + 1
    for i, path in enumerate(paths, start=start):
        img = load_img(path)
        img = rotate_by_angle(img, random_angle())

        word = file_to_word(path)
        fname = word_to_file(i, word)
        save_img(join(dst, fname), img)
