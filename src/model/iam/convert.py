from os.path import join

from model.common import input_dir
from model.iam.reader import IamReader
from model.utils.image import network_preprocess, save_img


def contains_reserved_character(word):
    """Checks if a word contains a reserved Windows path character."""
    reserved = set(["<", ">", ":", "/", "\\", "|", "?", "*"])
    word_set = set(list(word))

    return not reserved.isdisjoint(word_set)


def convert():
    """Converts the IAM dataset."""
    src = input_dir("Unconverted IAM dataset")
    dst = input_dir("Destination", create=True)

    reader = IamReader(src)
    for i, (data, img) in enumerate(reader.data_iter(), start=1):
        if contains_reserved_character(data.word):
            continue

        img = network_preprocess(img, data.word)

        fname = "{}-{}.png".format(str(i).zfill(6), data.word)
        save_img(join(dst, fname), img)
