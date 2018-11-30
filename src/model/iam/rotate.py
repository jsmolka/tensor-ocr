from glob import glob
from os.path import basename, join
from random import choice, randint

from model.common import input_dir, file_word
from model.utils.image import load_img, save_img, rotate_img


def rotate():
    """Converts the IAM dataset."""
    src = input_dir("Converted IAM dataset")
    
    files = glob(join(src, "*.png"))
    for i, fpath in enumerate(files, start=len(files) + 1):
        img = load_img(fpath)
        img = rotate_img(img, choice([-1, 1]) * randint(1, 5))

        word = file_word(fpath)
        save_img(join(src, "{}-{}.png".format(str(i).zfill(6), word)), img)
