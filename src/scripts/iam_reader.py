import cv2
import numpy as np
from glob import iglob
from os.path import basename, join, splitext


class WordLineData:
    def __init__(self, line):
        """Constructor."""
        self.path = ""
        self.ok = True
        self.gray = 0
        # Bounding box of x, y, width, height
        self.x = self.y = 0
        self.w = self.h = 0
        self.grammar = ""
        self.word = ""

        self.valid = self._parse_line(line)

    def _parse_line(self, line):
        """Parses the line."""
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
            # Convert the filename to match the one in words.txt
            # Ex: C:/dir/picture.png -> picture
            fn = basename(fl)
            fn = splitext(fn)[0]

            if fn not in self._data:
                continue

            try:
                img = cv2.imread(fl, cv2.IMREAD_GRAYSCALE)
                res = cv2.resize(img, dsize=(128, 64), interpolation=cv2.INTER_CUBIC)
            except BaseException:
                # Take care of corrupt files
                continue

            yield self._data[fn], res
    