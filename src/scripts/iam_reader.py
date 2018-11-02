import cv2
import glob
import numpy as np
from os.path import basename, exists, join, splitext

from iam_classes import WordLineData


class IamReader:
    def __init__(self, iam_dir):
        self._dir = iam_dir
        self._data = {}
        
        self._read_words()

    def _read_words(self):
        with open(join(self._dir, "ascii", "words.txt"), "r") as words:
            for line in words:
                if line.startswith("#"):
                    continue

                data = WordLineData(line[:-1])
                if data.valid:
                    self._data[data.path] = data

    @staticmethod
    def _file_iter(path, pattern, recursive=True):
        path = join(path, "**") if recursive else path
        return glob.iglob(join(path, pattern), recursive=recursive)

    @staticmethod
    def _filename(path):
        path = basename(path)
        return splitext(path)[0]

    def data_iter(self):
        for fl in self._file_iter(join(self._dir, "words"), "*.png"):
            fn = self._filename(fl)
            if fn in self._data:
                try:
                    img = cv2.imread(fl)
                    res = cv2.resize(img, dsize=(128, 64), interpolation=cv2.INTER_CUBIC)
                except BaseException:
                    continue

                yield self._data[fn], res
    