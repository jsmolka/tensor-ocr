import cv2
from os import makedirs
from os.path import exists, join

from iam_reader import IamReader


def main():
    """Main function."""
    # src assumes the follow dir structure:
    # src/
    # ├── ascii/
    # │   └── words.txt
    # └── words/
    src = input("Source dir: ")
    dst = input("Destination dir: ")
    if not exists(dst):
        makedirs(dst)

    reader = IamReader(src)
    
    i = 1
    for data, img in reader.data_iter():
        fn = "{}-{}.png".format(str(i).zfill(6), data.word)
        cv2.imwrite(join(dst, fn), img)
        i += 1


if __name__ == "__main__":
    main()