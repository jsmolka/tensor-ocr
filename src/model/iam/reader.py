from glob import iglob
from os.path import basename, join, splitext

from model.utils.image import load_img


class ImageData:
    def __init__(self, line):
        """Constructor."""
        self.path = ""
        self.ok = True
        self.gray = 0
        self.x = self.y = 0
        self.w = self.h = 0
        self.grammar = ""
        self.word = ""

        self.valid = self.parse_line(line)

    def parse_line(self, line):
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
        """
        Constructor.
        
        The source directory needs to have the following structure.
        src/
        ├── ascii/
        │   └── words.txt
        └── words/
        """
        self.src = src
        self.data = {}

        self.parse_words()

    def parse_words(self):
        """Parses words.txt and writes it into a dictionary."""
        with open(join(self.src, "ascii", "words.txt"), "r") as words:
            for line in words:
                if line.startswith("#"):
                    continue

                data = ImageData(line[:-1])
                if data.valid:
                    self.data[data.path] = data

    def data_iter(self):
        """Creates an iterator for data and image."""
        pattern = join(self.src, "words", "**", "*.png")
        for fpath in iglob(pattern, recursive=True):
            fname = basename(fpath)
            fname = splitext(fname)[0]

            if fname not in self.data:
                continue

            img = load_img(fpath)
            if (img is None or 0 in img.shape):
                continue

            yield self.data[fname], img
