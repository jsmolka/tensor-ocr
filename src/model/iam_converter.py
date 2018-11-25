import cv2
import numpy as np
import sys
from glob import iglob
from os import makedirs
from os.path import basename, exists, join, splitext

from constants import *


class WordLineData:
    def __init__(self, line):
        """Constructor."""
        self.path = ""
        self.ok = True
        self.gray = 0
        self.x = self.y = 0
        self.w = self.h = 0
        self.grammar = ""
        self.word = ""

        self.valid = self._parse_line(line)

    def _parse_line(self, line):
        """Parses a line."""
        parts = line.split(" ")
        try:
            self.path = parts[0]
            self.ok = parts[1] == "ok"
            self.gray = int(parts[2])
            self.x = int(parts[3])
            self.y = int(parts[4])
            self.w = int(parts[5])
            self.h = int(parts[6])
            self.grammar = parts[7]
            self.word = " ".join(parts[8:])
        except ValueError as e:
            return False

        return True


class IamReader:
    def __init__(self, src):
        """Constructor."""
        self._src = src
        self._data = {}

        self._parse_words()

    def _parse_words(self):
        """Parses words.txt and writes it into a dictionary."""
        with open(join(self._src, "ascii", "words.txt"), "r") as words:
            for line in words:
                if line.startswith("#"):
                    continue

                data = WordLineData(line[:-1])
                if data.valid:
                    self._data[data.path] = data

    def data_iter(self):
        """Creates an iterator for data and image."""
        globber = join(self._src, "words", "**", "*.png")
        for fl in iglob(globber, recursive=True):
            fn = basename(fl)
            fn = splitext(fn)[0]

            if fn not in self._data:
                continue

            img = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
            
            # Ignore corrupt images
            if (img is None or 0 in img.shape):
                continue

            yield self._data[fn], img


def resize(img, width, height):
    """Resizes an image using iterpolation."""
    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def resize_with_border(img, width, height):
    """Resizes an image by adding a white border."""
    h, w = img.shape
    bw = round((width - w) / 2)
    bh = round((height - h) / 2)

    return cv2.copyMakeBorder(img, bh, bh, bw, bw, cv2.BORDER_CONSTANT, value=[255, 255, 255])


def convert():
    """Converts the given IAM data."""
    # Assumes the follwing directory structure.
    # src/
    # ├── ascii/
    # │   └── words.txt
    # └── words/

    src = input("IAM source dir: ")
    if not exists(src):
        print("IAM source dir does not exist.")
        sys.exit(1)

    dst = input("Destination dir: ")
    if not exists(dst):
        makedirs(dst)

    reader = IamReader(src)
    
    for i, (data, img) in enumerate(reader.data_iter(), start=1):
        height, width = img.shape
        try:
            # Ignore words with reserved Windows characters.
            reserved = set(["<", ">", ":", "/", "\\", "|", "?", "*"])
            word_set = set(list(data.word))
            if not reserved.isdisjoint(word_set):
                continue

            # Resize short words using borders to prevent them from getting too
            # big. They still need to be resized afterwards because they might
            # not have the desired size.
            if len(data.word) <= 2 and height < img_h and width < img_w:
                img = resize_with_border(img, img_w, img_h)

            resized = resize(img, img_w, img_h)

            fn = "{}-{}.png".format(str(i).zfill(6), data.word)
            cv2.imwrite(join(dst, fn), resized)

        except Exception as e:
            print("Failed converting {} with error: {}".format(data.path, str(e)))


if __name__ == "__main__":
    convert()
